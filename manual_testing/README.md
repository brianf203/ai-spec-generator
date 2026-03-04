# Manual Testing for Paper

## Purpose

Compare **manual** vs **automatic** code addition to achieve 100% similarity. Manual uses less code % because we precisely add only what's missing (diff-based).

## Metric

- **auto_code_percent**: (lines in hybrid_code_additions / orig_lines) × 100
- **manual_code_percent**: (manual_added_lines / orig_lines) × 100
- **Expected**: manual_code_percent < auto_code_percent ✓

## Results (140 functions, 15 projects)

| Metric | Auto | Manual | Reduction |
|--------|------|--------|-----------|
| Avg code % | 59.6% | 27.2% | **32.3 pp** |
| Small (n=54) | 57.4% | 26.0% | 31.4 pp |
| Medium (n=36) | 50.9% | 25.3% | 25.6 pp |
| Large (n=50) | 68.1% | 29.9% | 38.2 pp |

## Workflow

1. **Extract**: `python extract_manual_test_data.py`
2. **Apply minimal (diff-based)**: `python apply_manual_minimal.py`
   - Computes lines in original but missing in regenerated
   - Simulates manual: add only what's strictly needed
3. **Aggregate**: `python aggregate_manual_results.py`
4. **Paper data**: `manual_test_results.json`

## Human Override

To manually override: edit `manual_test_worksheet.json` and set `manual_added_lines` for specific functions, then run `aggregate_manual_results.py`.
