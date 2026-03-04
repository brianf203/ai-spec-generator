#!/usr/bin/env python3
"""
Extract data for manual testing from benchmark spec_results.json.
Outputs a worksheet of functions that used hybrid (auto code additions),
so humans can determine the MINIMAL code needed to achieve 100% similarity.
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent

# Projects to include: 5 small, 5 medium, 5 large (the new ones)
PROJECTS = {
    "small": ["stack_project", "bitwise_project", "comparison_project", "range_project", "math_basic_project"],
    "medium": ["heap_project", "linked_project", "matrix_project", "permutation_project", "string_alg_project"],
    "large": ["metrics_project", "notification_project", "session_project", "rate_limiter_project", "file_project"],
}


def short_func_id(func_id: str) -> str:
    """Convert full path to short form: module::function"""
    if "::" in func_id:
        parts = func_id.split("::")
        module = Path(parts[0]).name
        return f"{module}::{parts[1]}"
    return func_id


def extract_from_project(size: str, name: str) -> list[dict]:
    """Extract hybrid-using functions from one project's spec_results.json."""
    path = BASE / f"output_{size}_benchmark" / name / "spec_results.json"
    if not path.exists():
        return []
    
    with open(path) as f:
        data = json.load(f)
    
    paper_data = data.get("analysis", {}).get("paper_data", {}).get("per_function", {})
    context = data.get("context", {})
    specifications = context.get("specifications", data.get("specifications", {}))
    similarity_results = context.get("similarity_results", {})
    
    rows = []
    for func_id, pd in paper_data.items():
        hybrid_pct = pd.get("hybrid_code_percent", 0)
        if hybrid_pct <= 0:
            continue
        
        # Get original and regenerated from similarity_results
        sr = similarity_results.get(func_id, {})
        orig = sr.get("original_code", "")
        regen = sr.get("regenerated_code", "")
        
        # Get hybrid_code_additions from spec
        spec_data = specifications.get(func_id, {})
        spec = spec_data.get("specification", {}) if isinstance(spec_data, dict) else spec_data
        auto_additions = spec.get("hybrid_code_additions", [])
        auto_additions = [a for a in auto_additions if str(a).strip()]
        
        orig_lines = len(orig.strip().split("\n")) if orig else 0
        auto_lines = sum(len(str(a).strip().split("\n")) for a in auto_additions)
        
        rows.append({
            "project": name,
            "size": size,
            "func_id": func_id,
            "short_id": short_func_id(func_id),
            "original_code": orig,
            "regenerated_code": regen,
            "auto_hybrid_additions": auto_additions,
            "orig_lines": orig_lines,
            "auto_added_lines": auto_lines,
            "auto_code_percent": hybrid_pct,
            "manual_added_lines": None,  # Human fills this
            "manual_code_additions": None,  # Human fills this (optional, for verification)
        })
    
    return rows


def main():
    all_rows = []
    for size, names in PROJECTS.items():
        for name in names:
            rows = extract_from_project(size, name)
            all_rows.extend(rows)
            print(f"  {size}/{name}: {len(rows)} functions using hybrid", file=sys.stderr)
    
    out_path = BASE / "manual_testing" / "manual_test_worksheet.json"
    out_path.parent.mkdir(exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    
    print(f"\nExtracted {len(all_rows)} functions to {out_path}", file=sys.stderr)
    return all_rows


if __name__ == "__main__":
    main()
