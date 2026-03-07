#!/usr/bin/env python3
"""Characterize JSON files and compute selectivity for user conditions.

Features:
- Per file: unique values per attribute and unique-count summary.
- Optional selectivity calculation with AND-combined conditions.
- Supports exact-match and numeric range-match conditions.

Input formats:
- JSON array of objects
- Single JSON object
- JSONL (one JSON object per line)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ExactCondition:
    key: str
    value: Any


@dataclass(frozen=True)
class RangeCondition:
    key: str
    min_value: float | None
    max_value: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile JSON files: per-attribute unique values/counts and optional "
            "selectivity for exact/range conditions."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input JSON files (.json or .jsonl).",
    )
    parser.add_argument(
        "--exact",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Exact-match condition. Repeatable. Examples: "
            "--exact category=shoe --exact in_stock=true --exact price=19.99"
        ),
    )
    parser.add_argument(
        "--range",
        action="append",
        default=[],
        metavar="KEY=MIN:MAX",
        help=(
            "Numeric range condition. Repeatable. MIN/MAX can be open ended. "
            "Examples: --range price=10:100 --range year=:2020 --range score=0.8:"
        ),
    )
    parser.add_argument(
        "--max-print-values",
        type=int,
        default=20,
        help=(
            "Maximum number of unique values printed per attribute. "
            "Use -1 to print all. Default: 20"
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write machine-readable profiling output JSON.",
    )
    return parser.parse_args()


def parse_scalar_literal(raw: str) -> Any:
    s = raw.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def parse_exact_conditions(raw_conditions: Sequence[str]) -> List[ExactCondition]:
    out: List[ExactCondition] = []
    for raw in raw_conditions:
        if "=" not in raw:
            raise ValueError(f"Invalid --exact '{raw}'. Expected KEY=VALUE.")
        key, value_raw = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --exact '{raw}'. Empty key.")
        out.append(ExactCondition(key=key, value=parse_scalar_literal(value_raw)))
    return out


def parse_range_conditions(raw_conditions: Sequence[str]) -> List[RangeCondition]:
    out: List[RangeCondition] = []
    for raw in raw_conditions:
        if "=" not in raw:
            raise ValueError(f"Invalid --range '{raw}'. Expected KEY=MIN:MAX.")
        key, spec = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --range '{raw}'. Empty key.")
        if ":" not in spec:
            raise ValueError(f"Invalid --range '{raw}'. Expected MIN:MAX.")
        min_raw, max_raw = spec.split(":", 1)
        min_val = float(min_raw) if min_raw.strip() else None
        max_val = float(max_raw) if max_raw.strip() else None
        out.append(RangeCondition(key=key, min_value=min_val, max_value=max_val))
    return out


def canonical_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return repr(value)


def flatten_record(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten_record(v, key))
            else:
                out[key] = v
    else:
        out["__value__"] = obj
    return out


def get_by_dotted_key(rec: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = rec
    for token in dotted_key.split("."):
        if not isinstance(cur, dict) or token not in cur:
            return None
        cur = cur[token]
    return cur


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            records: List[Dict[str, Any]] = []
            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"{path}: JSON array entry {i} is not an object; got {type(item).__name__}"
                    )
                records.append(item)
            return records
        if isinstance(parsed, dict):
            return [parsed]
        raise ValueError(f"{path}: top-level JSON must be an object or array of objects.")
    except json.JSONDecodeError:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"{path}: JSONL line {line_no} is not an object; got {type(obj).__name__}"
                    )
                records.append(obj)
        return records


def match_exact_conditions_all(rec: Dict[str, Any], conds: Sequence[ExactCondition]) -> bool:
    for cond in conds:
        value = get_by_dotted_key(rec, cond.key)
        if value != cond.value:
            return False
    return True


def match_exact_conditions_any(rec: Dict[str, Any], conds: Sequence[ExactCondition]) -> bool:
    if not conds:
        return True
    for cond in conds:
        value = get_by_dotted_key(rec, cond.key)
        if value == cond.value:
            return True
    return False


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def match_range_conditions(rec: Dict[str, Any], conds: Sequence[RangeCondition]) -> bool:
    for cond in conds:
        value = get_by_dotted_key(rec, cond.key)
        num = _as_float(value)
        if num is None:
            return False
        if cond.min_value is not None and num < cond.min_value:
            return False
        if cond.max_value is not None and num > cond.max_value:
            return False
    return True


def print_profile(
    *,
    path: Path,
    total: int,
    value_counters: Dict[str, Counter[str]],
    max_print_values: int,
    matched_and: int | None,
    matched_or: int | None,
    exact_conds: Sequence[ExactCondition],
    range_conds: Sequence[RangeCondition],
) -> None:
    print(f"\n=== {path} ===")
    print(f"Total records: {total}")

    for attr in sorted(value_counters.keys()):
        counter = value_counters[attr]
        unique_count = len(counter)
        print(f"- {attr}: unique_count={unique_count}")
        items = counter.items()
        if max_print_values >= 0:
            items = list(items)[:max_print_values]
        for v, cnt in items:
            print(f"    value={v} count={cnt}")
        if max_print_values >= 0 and unique_count > max_print_values:
            print(f"    ... ({unique_count - max_print_values} more values)")

    if matched_and is not None:
        ratio_and = (matched_and / total) if total > 0 else 0.0
        print("\nSelectivity:")
        if exact_conds:
            print("  Exact conditions:")
            for c in exact_conds:
                print(f"    {c.key} == {repr(c.value)}")
        if range_conds:
            print("  Range conditions:")
            for c in range_conds:
                print(f"    {c.min_value} <= {c.key} <= {c.max_value}")
        if len(exact_conds) >= 2 and matched_or is not None:
            ratio_or = (matched_or / total) if total > 0 else 0.0
            print(
                "  exact=AND + range=AND: "
                f"matched={matched_and} / total={total} -> selectivity={ratio_and:.8f}"
            )
            print(
                "  exact=OR  + range=AND: "
                f"matched={matched_or} / total={total} -> selectivity={ratio_or:.8f}"
            )
        else:
            print(
                "  matched="
                f"{matched_and} / total={total} -> selectivity={ratio_and:.8f}"
            )


def main() -> int:
    args = parse_args()
    if args.max_print_values < -1:
        raise ValueError("--max-print-values must be -1 or >= 0")

    exact_conds = parse_exact_conditions(args.exact)
    range_conds = parse_range_conditions(args.range)
    run_selectivity = bool(exact_conds or range_conds)

    all_results: Dict[str, Any] = {"files": []}

    for path in args.inputs:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

        records = list(iter_records(path))
        value_counters: Dict[str, Counter[str]] = defaultdict(Counter)
        matched_and = 0
        matched_or = 0

        for rec in records:
            flat = flatten_record(rec)
            for attr, value in flat.items():
                value_counters[attr][canonical_value(value)] += 1

            if run_selectivity:
                range_ok = match_range_conditions(rec, range_conds)
                exact_all_ok = match_exact_conditions_all(rec, exact_conds)
                exact_any_ok = match_exact_conditions_any(rec, exact_conds)

                if range_ok and exact_all_ok:
                    matched_and += 1
                if range_ok and exact_any_ok:
                    matched_or += 1

        total = len(records)
        print_profile(
            path=path,
            total=total,
            value_counters=value_counters,
            max_print_values=args.max_print_values,
            matched_and=matched_and if run_selectivity else None,
            matched_or=matched_or if run_selectivity else None,
            exact_conds=exact_conds,
            range_conds=range_conds,
        )

        file_json: Dict[str, Any] = {
            "path": str(path),
            "total_records": total,
            "attributes": {},
        }
        for attr in sorted(value_counters.keys()):
            counter = value_counters[attr]
            file_json["attributes"][attr] = {
                "unique_count": len(counter),
                "value_counts": dict(counter),
            }

        if run_selectivity:
            ratio_and = (matched_and / total) if total > 0 else 0.0
            ratio_or = (matched_or / total) if total > 0 else 0.0
            selectivity_json: Dict[str, Any] = {
                "exact_conditions": [{"key": c.key, "value": c.value} for c in exact_conds],
                "range_conditions": [
                    {"key": c.key, "min": c.min_value, "max": c.max_value} for c in range_conds
                ],
                "and": {
                    "matched": matched_and,
                    "total": total,
                    "ratio": ratio_and,
                },
            }
            if len(exact_conds) >= 2:
                selectivity_json["or"] = {
                    "matched": matched_or,
                    "total": total,
                    "ratio": ratio_or,
                }
            file_json["selectivity"] = selectivity_json
        all_results["files"].append(file_json)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON report: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
