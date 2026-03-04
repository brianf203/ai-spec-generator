#!/usr/bin/env python3
"""
Aggregate manual testing results for the paper.
Reads manual_test_worksheet.json (with manual_added_lines filled in),
computes manual vs auto code percent, and outputs paper-ready summary.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent


def main():
    path = BASE / "manual_test_worksheet.json"
    if not path.exists():
        print("Run extract_manual_test_data.py first.", file=sys.stderr)
        sys.exit(1)
    
    with open(path) as f:
        rows = json.load(f)
    
    # Filter to rows with manual_added_lines filled
    manual_rows = [r for r in rows if r.get("manual_added_lines") is not None]
    if not manual_rows:
        print("No manual results yet. Fill manual_added_lines in manual_test_worksheet.json", file=sys.stderr)
        sys.exit(1)
    
    # Compute metrics
    auto_pcts = []
    manual_pcts = []
    by_size = defaultdict(lambda: {"auto": [], "manual": []})
    by_project = defaultdict(lambda: {"auto": [], "manual": []})
    
    for r in manual_rows:
        orig_lines = r.get("orig_lines", 1) or 1
        auto_pct = r.get("auto_code_percent", 0)
        manual_lines = r.get("manual_added_lines", 0)
        manual_pct = (manual_lines / orig_lines * 100) if orig_lines > 0 else 0
        
        auto_pcts.append(auto_pct)
        manual_pcts.append(manual_pct)
        by_size[r["size"]]["auto"].append(auto_pct)
        by_size[r["size"]]["manual"].append(manual_pct)
        by_project[r["project"]]["auto"].append(auto_pct)
        by_project[r["project"]]["manual"].append(manual_pct)
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    # Output
    print("=" * 60)
    print("MANUAL vs AUTO: Code % to achieve 100% similarity")
    print("=" * 60)
    print(f"\nSample size: {len(manual_rows)} functions (of {len(rows)} total)")
    print(f"\nOverall:")
    print(f"  Auto:   avg={avg(auto_pcts):.1f}%  min={min(auto_pcts):.1f}%  max={max(auto_pcts):.1f}%")
    print(f"  Manual: avg={avg(manual_pcts):.1f}%  min={min(manual_pcts):.1f}%  max={max(manual_pcts):.1f}%")
    print(f"  Reduction: {avg(auto_pcts) - avg(manual_pcts):.1f} percentage points")
    
    print(f"\nBy size:")
    for size in ["small", "medium", "large"]:
        if size in by_size:
            a, m = by_size[size]["auto"], by_size[size]["manual"]
            print(f"  {size}: auto avg={avg(a):.1f}%  manual avg={avg(m):.1f}%  (n={len(a)})")
    
    print(f"\nBy project:")
    for proj in sorted(by_project.keys()):
        a, m = by_project[proj]["auto"], by_project[proj]["manual"]
        print(f"  {proj}: auto={avg(a):.1f}%  manual={avg(m):.1f}%  (n={len(a)})")
    
    # Save paper-ready JSON
    out = {
        "manual_sample_size": len(manual_rows),
        "total_hybrid_functions": len(rows),
        "auto_avg_code_percent": round(avg(auto_pcts), 2),
        "manual_avg_code_percent": round(avg(manual_pcts), 2),
        "reduction_pp": round(avg(auto_pcts) - avg(manual_pcts), 2),
        "by_size": {
            s: {"auto_avg": round(avg(d["auto"]), 2), "manual_avg": round(avg(d["manual"]), 2), "n": len(d["auto"])}
            for s, d in by_size.items()
        },
        "per_function": [
            {
                "short_id": r["short_id"],
                "project": r["project"],
                "size": r["size"],
                "auto_code_percent": r["auto_code_percent"],
                "manual_code_percent": round((r.get("manual_added_lines", 0) / (r.get("orig_lines") or 1)) * 100, 2),
            }
            for r in manual_rows
        ],
    }
    out_path = BASE / "manual_test_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
