"""
Run repository pytest files as a behavioral oracle (Phase B).

When AST ``TestLoader`` extraction yields zero cases, the pipeline can subprocess
``pytest`` on discovered test modules, compare pass/fail on original vs patched
source (one qualified function replaced with regenerated code).
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from utils.test_loader import TestLoader


@dataclass
class PytestRunSummary:
    exit_code: int
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    failed_nodeids: List[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0


def discover_pytest_paths(source_file: str, project_path: str) -> List[str]:
    """Return absolute paths to pytest modules associated with a source file."""
    paths: List[str] = []
    primary = TestLoader.find_test_file(source_file, project_path)
    if primary:
        paths.append(os.path.abspath(primary))

    source_path = Path(source_file)
    project = Path(project_path).resolve()
    stem = source_path.stem
    rel_parent = source_path.parent
    try:
        rel_parent = rel_parent.resolve().relative_to(project)
    except ValueError:
        rel_parent = Path(".")

    candidates = [
        project / "tests" / f"test_{stem}.py",
        project / "tests" / "claude_generated" / f"test_{stem}.py",
        project / rel_parent / "tests" / f"test_{stem}.py",
        project / rel_parent / "tests" / "claude_generated" / f"test_{stem}.py",
    ]
    for c in candidates:
        ap = str(c.resolve()) if c.exists() else None
        if ap and ap not in paths:
            paths.append(ap)
    return paths


def _parse_junit(path: Path) -> PytestRunSummary:
    if not path.is_file():
        return PytestRunSummary(exit_code=1, total=0, passed=0, failed=0, errors=1, skipped=0)
    root = ET.parse(path).getroot()
    failed_ids: List[str] = []
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = root.findall("testsuite")

    total = passed = failed = errors = skipped = 0
    for suite in suites:
        total += int(suite.attrib.get("tests", 0))
        failed += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        skipped += int(suite.attrib.get("skipped", 0))
        for case in suite.findall("testcase"):
            cls = case.find("failure")
            err = case.find("error")
            if cls is not None or err is not None:
                name = case.attrib.get("classname", "")
                method = case.attrib.get("name", "")
                failed_ids.append(f"{name}::{method}" if name else method)

    passed = max(0, total - failed - errors - skipped)
    exit_code = 0 if failed == 0 and errors == 0 else 1
    return PytestRunSummary(
        exit_code=exit_code,
        total=total,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        failed_nodeids=failed_ids,
    )


def run_pytest(
    project_path: str,
    test_paths: List[str],
    *,
    timeout_sec: int = 300,
) -> PytestRunSummary:
    """Execute pytest on ``test_paths`` with cwd=project root."""
    import tempfile

    if not test_paths:
        return PytestRunSummary(exit_code=1, total=0, passed=0, failed=0, errors=1, skipped=0)

    project = Path(project_path).resolve()
    rootdir = Path(test_paths[0]).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        junit = Path(td) / "junit.xml"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--config-file=/dev/null",
            f"--rootdir={rootdir}",
            "-q",
            "--tb=no",
            f"--junitxml={junit}",
            *test_paths,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(project),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        summary = _parse_junit(junit)
        if summary.total == 0 and proc.returncode != 0:
            # Fallback: parse "N passed" from stdout
            m = re.search(r"(\d+)\s+passed", proc.stdout or "")
            if m:
                n = int(m.group(1))
                summary = PytestRunSummary(
                    exit_code=proc.returncode,
                    total=n,
                    passed=n if proc.returncode == 0 else 0,
                    failed=0 if proc.returncode == 0 else 1,
                    errors=0,
                    skipped=0,
                )
            else:
                summary.exit_code = proc.returncode
        else:
            summary.exit_code = proc.returncode
        return summary


def _find_function_span(
    tree: ast.AST, qualified_key: str, lines: List[str]
) -> Tuple[int, int]:
    """Return (start_line, end_line) 1-based inclusive for decorators+def body."""
    parts = qualified_key.split(".")
    func_name = parts[-1]
    class_names = parts[:-1]

    def match_in_class(class_node: ast.ClassDef) -> Optional[Tuple[int, int]]:
        if class_names and class_node.name != class_names[0]:
            return None
        if len(class_names) > 1:
            for node in class_node.body:
                if isinstance(node, ast.ClassDef) and node.name == class_names[1]:
                    return match_in_class(node)
            return None
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                start = node.lineno
                if hasattr(node, "decorator_list") and node.decorator_list:
                    start = min(d.lineno for d in node.decorator_list)
                end = node.end_lineno or node.lineno
                return start, end
        return None

    if class_names:
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                span = match_in_class(node)
                if span:
                    return span
        raise ValueError(f"Could not locate {qualified_key!r} in module")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            start = node.lineno
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list)
            end = node.end_lineno or node.lineno
            return start, end
    raise ValueError(f"Could not locate {qualified_key!r} in module")


@contextmanager
def patch_source_callable(
    source_file: str,
    qualified_key: str,
    replacement_source: str,
) -> Iterator[None]:
    """Replace one function/method in ``source_file``; restore on exit."""
    path = Path(source_file)
    original_text = path.read_text(encoding="utf-8")
    lines = original_text.splitlines(keepends=True)
    tree = ast.parse(original_text)
    start, end = _find_function_span(tree, qualified_key, lines)
    start_idx = start - 1
    end_idx = end  # end_lineno inclusive -> slice end exclusive

    orig_line = lines[start_idx]
    indent = len(orig_line) - len(orig_line.lstrip())
    repl = textwrap.dedent(replacement_source).strip("\n")
    if not repl.lstrip().startswith(("def ", "async def ")):
        raise ValueError(
            f"Replacement for {qualified_key} must include a def/async def header"
        )
    indented = textwrap.indent(repl, " " * indent) + "\n"
    if not indented.endswith("\n"):
        indented += "\n"

    patched = "".join(lines[:start_idx]) + indented + "".join(lines[end_idx:])
    path.write_text(patched, encoding="utf-8")
    try:
        yield
    finally:
        path.write_text(original_text, encoding="utf-8")


def _behavioral_failures(
    baseline: PytestRunSummary, regen: PytestRunSummary
) -> List[Dict[str, Any]]:
    """Tests whose pass/fail status differs between baseline and patched run."""
    if baseline.total == 0:
        return [{"test": {"test_name": "pytest_suite"}, "reason": "no tests collected"}]

    base_fail = set(baseline.failed_nodeids)
    regen_fail = set(regen.failed_nodeids)
    mismatches = (base_fail ^ regen_fail) | (
        base_fail & regen_fail
    )  # include regen-only failures when baseline passed all

    if baseline.all_passed and regen.all_passed:
        return []

    failures: List[Dict[str, Any]] = []
    if baseline.all_passed and not regen.all_passed:
        for nid in regen.failed_nodeids[:25]:
            failures.append(
                {
                    "test": {"test_name": nid},
                    "reason": "regenerated_code_failed_pytest",
                    "regenerated_output": "fail",
                    "original_output": "pass",
                }
            )
        # pad count to reflect failed test count for similarity ratio
        extra = regen.failed - len(failures)
        for i in range(extra):
            failures.append(
                {
                    "test": {"test_name": f"pytest_failed_{i}"},
                    "reason": "regenerated_code_failed_pytest",
                }
            )
        return failures

    if not baseline.all_passed:
        failures.append(
            {
                "test": {"test_name": "pytest_baseline"},
                "reason": "baseline_pytest_not_all_pass",
            }
        )
    return failures


def run_pytest_harness_for_function(
    *,
    project_path: str,
    source_file: str,
    qualified_key: str,
    pytest_paths: List[str],
    original_source: str,
    regenerated_source: str,
    baseline_cache: Dict[str, PytestRunSummary],
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    cache_key = f"{source_file}::{'|'.join(sorted(pytest_paths))}"
    if cache_key not in baseline_cache:
        baseline_cache[cache_key] = run_pytest(
            project_path, pytest_paths, timeout_sec=timeout_sec
        )
    baseline = baseline_cache[cache_key]

    regen_summary = baseline
    if regenerated_source.strip() and regenerated_source.strip() != original_source.strip():
        try:
            with patch_source_callable(source_file, qualified_key, regenerated_source):
                regen_summary = run_pytest(
                    project_path, pytest_paths, timeout_sec=timeout_sec
                )
        except Exception as exc:
            regen_summary = PytestRunSummary(
                exit_code=1,
                total=baseline.total,
                passed=0,
                failed=baseline.total or 1,
                errors=0,
                skipped=0,
                failed_nodeids=[f"patch_error:{exc}"],
            )

    total = max(baseline.total, regen_summary.total, 1)
    orig_passed = baseline.passed if baseline.all_passed else baseline.passed
    regen_passed = regen_summary.passed if regen_summary.all_passed else regen_summary.passed
    failures = _behavioral_failures(baseline, regen_summary)

    if baseline.all_passed and regen_summary.all_passed:
        behavioral_match = True
        regen_passed = total
        orig_passed = total
    elif baseline.all_passed:
        behavioral_match = regen_summary.all_passed
        orig_passed = total
        regen_passed = total - len(failures)
    else:
        behavioral_match = False
        orig_passed = baseline.passed
        regen_passed = regen_summary.passed

    return {
        "total_tests": total,
        "original_passed": orig_passed,
        "original_failed": total - orig_passed,
        "regenerated_passed": regen_passed,
        "regenerated_failed": total - regen_passed,
        "failures": failures,
        "behavioral_match": behavioral_match,
        "branch_coverage": 0.0,
        "coverage_complete": False,
        "harness_mode": "pytest",
        "pytest_paths": pytest_paths,
        "pytest_baseline_all_passed": baseline.all_passed,
        "pytest_regen_all_passed": regen_summary.all_passed,
        "pytest_baseline_failed_nodeids": baseline.failed_nodeids[:10],
        "pytest_regen_failed_nodeids": regen_summary.failed_nodeids[:10],
    }
