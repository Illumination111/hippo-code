"""Dependency-free BioCache state proxy for HiPPO recurrent cells."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Callable, Mapping

SCHEMA_VERSION = "biocache-adapt/v1"
MANIFEST_SCHEMA = {
    "$id": "biocache-adapt/manifest/v1",
    "type": "object",
    "required": ["schema_version", "adapter", "repository", "implementation", "capabilities", "budget"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "adapter": {"type": "string"},
        "repository": {"type": "string"},
        "implementation": {"type": "string"},
        "capabilities": {"type": "array", "items": {"type": "string"}},
        "budget": {"type": "object"},
    },
}
DELAYED_RECALL_SCHEMA = {
    "$id": "biocache-adapt/jsonl/delayed-recall/v1",
    "type": "object",
    "required": ["schema_version", "record_type", "sample_id", "write_step", "query_step", "expected"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "record_type": {"const": "delayed_recall"},
        "sample_id": {"type": "string"},
        "write_step": {"type": "integer", "minimum": 0},
        "query_step": {"type": "integer", "minimum": 0},
        "expected": {},
        "prediction": {},
        "metadata": {"type": "object"},
    },
}
VNBENCH_FEATURE_SCHEMA = {
    "$id": "biocache-adapt/jsonl/vnbench-feature/v1",
    "type": "object",
    "required": ["schema_version", "record_type", "sample_id", "feature_name", "feature"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "record_type": {"const": "vnbench_feature"},
        "sample_id": {"type": "string"},
        "feature_name": {"type": "string"},
        "feature": {},
        "frame_index": {"type": "integer", "minimum": 0},
        "timestamp_ms": {"type": "number", "minimum": 0},
        "metadata": {"type": "object"},
    },
}


@dataclass(frozen=True)
class BudgetConfig:
    max_persistent_bytes: int | None = None
    max_static_extra_bytes: int | None = None
    max_temporary_bytes: int | None = None
    max_total_bytes: int | None = None
    max_update_ms: float | None = None
    max_query_ms: float | None = None


@dataclass
class HiPPOPersistentState:
    h: Any = None
    memory: Any = None
    time_step: Any = 0

    @classmethod
    def from_tuple(cls, state: Any) -> "HiPPOPersistentState":
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise TypeError("HiPPO state must be a (h, memory, time_step) triple")
        return cls(state[0], state[1], state[2])

    def as_tuple(self) -> tuple[Any, Any, Any]:
        return self.h, self.memory, self.time_step


class BudgetExceeded(RuntimeError):
    pass


def recursive_nbytes(value: Any, _seen: set[int] | None = None) -> int:
    if value is None:
        return 0
    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    numel, element_size = getattr(value, "numel", None), getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    if is_dataclass(value) and not isinstance(value, type):
        return sys.getsizeof(value) + sum(
            recursive_nbytes(getattr(value, field.name), seen) for field in fields(value)
        )
    if isinstance(value, Mapping):
        return sys.getsizeof(value) + sum(
            recursive_nbytes(key, seen) + recursive_nbytes(item, seen) for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return sys.getsizeof(value) + sum(recursive_nbytes(item, seen) for item in value)
    return sys.getsizeof(value)


def validate_record(record: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    missing = [name for name in schema.get("required", ()) if name not in record]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    for name, rule in schema.get("properties", {}).items():
        if name in record and "const" in rule and record[name] != rule["const"]:
            raise ValueError(f"{name} must equal {rule['const']!r}")


class HiPPOBioCacheAdapter:
    """Proxy for a HiPPO cell state; no VLM or retrieval layer is implied."""

    adapter_name = "hippo-code"
    repository = "hippo-code"
    implementation = "deprecated-official-research-code"

    def __init__(
        self,
        *,
        cell: Any = None,
        h: Any = None,
        memory: Any = None,
        time_step: Any = 0,
        static_extra_state: Any = None,
        budget: BudgetConfig | None = None,
        update_fn: Callable[[Any, HiPPOPersistentState, Any], Any] | None = None,
        query_fn: Callable[[Any, HiPPOPersistentState, Any], Any] | None = None,
    ) -> None:
        self.cell = cell
        self.budget = budget or BudgetConfig()
        self._initial_state = HiPPOPersistentState(h, memory, time_step)
        self._persistent_state = self._initial_state
        self._static_extra_state = {"cell": cell} if static_extra_state is None else static_extra_state
        self._temporary_state = None
        self._update_fn, self._query_fn = update_fn, query_fn
        self.timings = {
            "update": {"calls": 0, "total_ms": 0.0, "last_ms": 0.0},
            "query": {"calls": 0, "total_ms": 0.0, "last_ms": 0.0},
        }
        self._check_budget()

    @classmethod
    def from_cell_state(
        cls, cell: Any, state: tuple[Any, Any, Any], **kwargs: Any
    ) -> "HiPPOBioCacheAdapter":
        packed = HiPPOPersistentState.from_tuple(state)
        return cls(
            cell=cell,
            h=packed.h,
            memory=packed.memory,
            time_step=packed.time_step,
            **kwargs,
        )

    @property
    def persistent_state(self) -> HiPPOPersistentState:
        return self._persistent_state

    @property
    def static_extra_state(self) -> Any:
        return self._static_extra_state

    @property
    def temporary_state(self) -> Any:
        return self._temporary_state

    def reset(
        self,
        *,
        persistent_state: HiPPOPersistentState | None = None,
        static_extra_state: Any = None,
    ) -> HiPPOPersistentState:
        self._persistent_state = persistent_state or self._initial_state
        if static_extra_state is not None:
            self._static_extra_state = static_extra_state
        self._temporary_state = None
        self._check_budget()
        return self._persistent_state

    def update(
        self,
        observation: Any = None,
        *,
        persistent_state: HiPPOPersistentState | None = None,
        temporary_state: Any = None,
    ) -> HiPPOPersistentState:
        started = time.perf_counter()
        output = None
        if persistent_state is not None:
            self._persistent_state = persistent_state
        elif self._update_fn is not None:
            result = self._update_fn(observation, self._persistent_state, temporary_state)
            output, state = result if isinstance(result, tuple) and len(result) == 2 else (None, result)
            self._persistent_state = (
                state if isinstance(state, HiPPOPersistentState) else HiPPOPersistentState.from_tuple(state)
            )
        elif self.cell is not None and observation is not None:
            output, state = self.cell.forward(observation, self._persistent_state.as_tuple())
            self._persistent_state = HiPPOPersistentState.from_tuple(state)
        self._temporary_state = {"caller": temporary_state, "last_output": output}
        self._record_timing("update", started)
        self._check_budget()
        return self._persistent_state

    def query(self, request: Any = None, *, temporary_state: Any = None) -> Any:
        started = time.perf_counter()
        previous_output = (
            self._temporary_state.get("last_output")
            if isinstance(self._temporary_state, Mapping)
            else None
        )
        self._temporary_state = {"caller": temporary_state, "last_output": previous_output}
        if self._query_fn is not None:
            result = self._query_fn(request, self._persistent_state, temporary_state)
        elif previous_output is not None:
            result = previous_output
        else:
            result = self._persistent_state.h
        self._record_timing("query", started)
        self._check_budget()
        return result

    def state_bytes(self) -> dict[str, int]:
        sizes = {
            "persistent_state": recursive_nbytes(self._persistent_state),
            "static_extra_state": recursive_nbytes(self._static_extra_state),
            "temporary_state": recursive_nbytes(self._temporary_state),
        }
        sizes["total"] = sum(sizes.values())
        return sizes

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "adapter": self.adapter_name,
            "repository": self.repository,
            "implementation": self.implementation,
            "capabilities": [
                "proxy_only",
                "h",
                "memory",
                "time_step",
                "recursive_byte_accounting",
                "timing",
            ],
            "budget": asdict(self.budget),
        }

    def _record_timing(self, operation: str, started: float) -> None:
        elapsed = (time.perf_counter() - started) * 1000.0
        stats = self.timings[operation]
        stats["calls"] += 1
        stats["last_ms"] = elapsed
        stats["total_ms"] += elapsed
        limit = getattr(self.budget, f"max_{operation}_ms")
        if limit is not None and elapsed > limit:
            raise BudgetExceeded(f"{operation} took {elapsed:.3f} ms, budget is {limit:.3f} ms")

    def _check_budget(self) -> None:
        sizes = self.state_bytes()
        limits = {
            "persistent_state": self.budget.max_persistent_bytes,
            "static_extra_state": self.budget.max_static_extra_bytes,
            "temporary_state": self.budget.max_temporary_bytes,
            "total": self.budget.max_total_bytes,
        }
        for name, limit in limits.items():
            if limit is not None and sizes[name] > limit:
                raise BudgetExceeded(f"{name} uses {sizes[name]} bytes, budget is {limit}")


def dry_run() -> dict[str, Any]:
    adapter = HiPPOBioCacheAdapter(h=b"hidden", memory=b"memory", time_step=3)
    adapter.reset()
    adapter.update(temporary_state={"phase": "write"})
    adapter.query({"phase": "recall"}, temporary_state={"phase": "read"})
    delayed = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "delayed_recall",
        "sample_id": "dry-run",
        "write_step": 0,
        "query_step": 1,
        "expected": "sentinel",
    }
    feature = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "vnbench_feature",
        "sample_id": "dry-run",
        "feature_name": "proxy",
        "feature": [0.0],
    }
    validate_record(adapter.manifest(), MANIFEST_SCHEMA)
    validate_record(delayed, DELAYED_RECALL_SCHEMA)
    validate_record(feature, VNBENCH_FEATURE_SCHEMA)
    return {
        "ok": True,
        "manifest": adapter.manifest(),
        "state_bytes": adapter.state_bytes(),
        "timings": adapter.timings,
        "jsonl_examples": [delayed, feature],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="HiPPO BioCache state proxy (no model execution)")
    parser.add_argument("--dry-run", action="store_true", help="validate schemas and proxy state only")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.print_help()
        return 0
    print(json.dumps(dry_run(), indent=2 if args.pretty else None, sort_keys=True))
    return 0


__all__ = [
    "BudgetConfig",
    "BudgetExceeded",
    "DELAYED_RECALL_SCHEMA",
    "HiPPOBioCacheAdapter",
    "HiPPOPersistentState",
    "MANIFEST_SCHEMA",
    "SCHEMA_VERSION",
    "VNBENCH_FEATURE_SCHEMA",
    "dry_run",
    "recursive_nbytes",
    "validate_record",
]
