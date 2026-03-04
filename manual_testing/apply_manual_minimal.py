#!/usr/bin/env python3
"""
Compute minimal code addition by diffing original vs regenerated.
Simulates manual approach: add only what's in original but missing in regenerated.
This gives a lower bound on manual_code_percent since a human could do at least this well.
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).parent


def normalize_for_compare(line: str) -> str:
    """Normalize line for comparison (strip, collapse whitespace)."""
    return " ".join(line.strip().split())


def lines_in_original_not_in_regenerated(orig: str, regen: str) -> int:
    """
    Count lines from original that are missing in regenerated.
    Uses normalized line comparison. Docstrings and logic lines not in regen count.
    """
    if not orig or not regen:
        return 0
    orig_lines = [l for l in orig.strip().split("\n") if l.strip()]
    regen_lines = [l for l in regen.strip().split("\n") if l.strip()]
    if not orig_lines:
        return 0
    regen_norm = {normalize_for_compare(l) for l in regen_lines}
    missing = 0
    for line in orig_lines:
        norm = normalize_for_compare(line)
        if norm in regen_norm:
            continue
        # Line is in original but not regenerated - count it
        missing += 1
    return missing


def main():
    path = BASE / "manual_test_worksheet.json"
    with open(path) as f:
        rows = json.load(f)
    
    for r in rows:
        orig = r.get("original_code", "")
        regen = r.get("regenerated_code", "")
        if not orig or not regen:
            r["manual_added_lines"] = 0
            continue
        # Minimal = lines in original that aren't in regenerated
        missing = lines_in_original_not_in_regenerated(orig, regen)
        r["manual_added_lines"] = missing
    
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    
    print(f"Updated {len(rows)} rows with manual_added_lines (diff-based minimal)")
    print("Run aggregate_manual_results.py to see results.")


if __name__ == "__main__":
    main()
