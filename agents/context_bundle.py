"""
Principled bounded-context packaging for spec / codegen (v2).

Builds deterministic, budgeted excerpts around a target callable:
  1) same-module callees (static name→definition match inside the analyzed file map)
  2) enclosing class excerpt (__init__ + non-target method stubs + truncated class skeleton)
  3) multi-hop expansion within a scoped subtree of the repo (ancestor directory roots)
  4) optional name-based retrieval ("RAG") over an index restricted to scope

Every chunk records {rule, key, chars, fingerprint} so inclusion is reproducible.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple


FuncNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass
class InclusionRecord:
    rule: str
    key: str
    chars: int
    sha256_preview: str
    excerpt_line_start: Optional[int] = None
    excerpt_line_end: Optional[int] = None


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:16]


def truncate_utf8(s: str, max_bytes: int) -> str:
    """Truncate string so UTF-8 encoding length is ≤ max_bytes."""
    if max_bytes <= 0:
        return ""
    b = s.encode("utf-8", errors="replace")
    if len(b) <= max_bytes:
        return s
    b = b[:max_bytes]
    while b:
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            b = b[:-1]
    return ""


_SKIP_BUILTIN_CALL_NAMES = frozenset(
    {
        "print",
        "super",
        "len",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "range",
        "enumerate",
        "isinstance",
        "getattr",
        "setattr",
        "hasattr",
        "type",
        "open",
        "min",
        "max",
        "abs",
        "sum",
        "any",
        "all",
        "iter",
        "next",
        "ord",
        "chr",
        "repr",
        "sorted",
        "reversed",
    }
)


def _find_qualified_node(tree: ast.Module, parts: Sequence[str]) -> Optional[FuncNode]:
    if not parts:
        return None
    if len(parts) == 1:
        name = parts[0]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        return None

    *class_names, method_name = parts
    cur_body = tree.body
    for cname in class_names:
        cls_node = None
        for node in cur_body:
            if isinstance(node, ast.ClassDef) and node.name == cname:
                cls_node = node
                break
        if cls_node is None:
            return None
        cur_body = cls_node.body

    for node in cur_body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return node
    return None


def extract_call_names(fn_node: FuncNode) -> FrozenSet[str]:
    """Unresolved call keys: bare Name.id and Attribute.tail for Attribute chains."""
    out: Set[str] = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name):
            out.add(f.id)
        elif isinstance(f, ast.Attribute):
            out.add(f.attr)
    return frozenset(out)


def scope_roots_anchor(
    project_root: Path, anchor_file: Path, parent_levels: int
) -> List[Path]:
    """
    Roots = [directory of ``anchor_file``], optionally plus parent dirs up to
    ``parent_levels`` ascents, all forced to remain under ``project_root``.

    ``parent_levels`` = 0 → only the anchor file's directory.
    A candidate file is allowed if it lies under the union of these directory subtrees.
    """
    proj = project_root.resolve()
    cur = anchor_file.resolve().parent
    roots: List[Path] = []
    ascents_remaining = max(0, int(parent_levels))
    roots.append(cur.resolve())
    while ascents_remaining > 0:
        par = cur.parent.resolve()
        if par == cur:
            break
        try:
            par.relative_to(proj)
        except ValueError:
            break
        roots.append(par)
        ascents_remaining -= 1
        cur = par
    return roots


def path_allowed(paths: Iterable[Path], candidate: Path) -> bool:
    c = candidate.resolve()
    for r in paths:
        try:
            c.relative_to(r.resolve())
            return True
        except ValueError:
            continue
    return False


def find_function_matching_name(
    functions_map: Dict[str, Dict[str, Any]],
    callee_name: str,
    exclude_keys: FrozenSet[str],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Pick one definition: prioritize exact qualified tail match, smallest line."""
    cand: List[Tuple[str, int, Dict[str, Any]]] = []
    for qk, finfo in functions_map.items():
        if qk in exclude_keys:
            continue
        tail = qk.split(".")[-1]
        if tail == callee_name or qk == callee_name:
            ln = int(finfo.get("line_number") or 0)
            cand.append((qk, ln, finfo))
    if not cand:
        return None
    cand.sort(key=lambda x: (x[1], x[0]))
    qk, _ln, fi = cand[0]
    return qk, fi


def build_symbol_index_for_scope(
    python_files: List[str],
    allowed_roots: List[Path],
    project_root: Path,
    analyzed_files: Dict[str, Any],
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Lowercased lookup key → list of (file_path, qualified_key, excerpt_first_line_chars)
    for duplicate-name diagnostics (stable sort uses path+qk).

    Prefer analyzed_files[\"functions\"] when present; parse otherwise.
    """
    idx: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for fp in sorted(python_files):
        pf = Path(fp)
        if not path_allowed(allowed_roots, pf):
            continue
        fp_s = str(pf.resolve())
        fa = analyzed_files.get(fp_s) or analyzed_files.get(fp)
        funcs_map: Dict[str, Dict[str, Any]] = {}
        if fa:
            fm = fa.get("functions") or {}
            funcs_map = fm
        else:
            try:
                content = pf.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(content)
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        key = node.name
                        seg = ast.get_source_segment(content, node) or ""
                        funcs_map[key] = {
                            "source": seg,
                            "line_number": node.lineno,
                        }
                    elif isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                key = f"{node.name}.{item.name}"
                                seg = ast.get_source_segment(content, item) or ""
                                funcs_map[key] = {
                                    "source": seg,
                                    "line_number": item.lineno,
                                }
            except (OSError, SyntaxError):
                continue

        for qk, fi in funcs_map.items():
            blob = fi.get("source") or ""
            tail = qk.split(".")[-1].lower()
            idx[tail.lower()].append((fp_s, qk, blob[:120].replace("\n", " ") if blob else ""))
    for vs in idx.values():
        vs.sort(key=lambda t: (t[0], t[1]))
    return idx


@dataclass
class PackedBundle:
    text: str
    manifest: List[Dict[str, Any]] = field(default_factory=list)


def format_chunk(rule: str, key: str, content: str) -> Tuple[str, InclusionRecord]:
    header = (
        f"### [{rule}] {key}\n"
        "```python\n"
    )
    body = content.rstrip() + "\n"
    footer = "```\n"
    snippet = header + body + footer
    lines = len(content.splitlines())
    rec = InclusionRecord(
        rule=rule,
        key=key,
        chars=len(snippet.encode("utf-8", errors="replace")),
        sha256_preview=_sha(content),
        excerpt_line_start=1,
        excerpt_line_end=lines,
    )
    return snippet, rec


class BoundedContextBundler:
    def __init__(self, config: Dict[str, Any]):
        c = config
        self.budget_chars = int(c.get("context_budget_chars", 28_672))
        self.k_hop = int(c.get("context_k_hop", 2))
        self.scope_parent_levels = int(c.get("context_scope_parent_levels", 1))
        self.enable_rag = bool(c.get("context_enable_rag", False))
        self.rag_top_k = int(c.get("context_rag_top_k", 4))
        self.class_overhead_budget_frac = float(c.get("context_class_budget_frac", 0.22))

    def pack(
        self,
        *,
        project_root: str,
        anchor_file: str,
        qualified_key: str,
        target_source: str,
        analyzed_files: Dict[str, Any],
        python_files: List[str],
        dependencies: Optional[List[str]] = None,
    ) -> PackedBundle:
        proj = Path(project_root).resolve()
        anchor_fp = Path(anchor_file).resolve()

        deps_list = dependencies or []

        exclude_target = frozenset({qualified_key})
        consumed = 0

        chunks: List[str] = []
        manifest: List[Dict[str, Any]] = []

        def add(rule: str, key: str, content: str) -> bool:
            nonlocal consumed
            if consumed >= self.budget_chars:
                return False
            if not content or not content.strip():
                return False
            leftover = self.budget_chars - consumed
            if leftover < 160:
                return False

            clipped = content
            if len(clipped.encode("utf-8", errors="replace")) > leftover - 160:
                # Hard clip by unicode string length proxy
                clipped = clipped[: max(0, leftover - 480)] + "\n# [... truncated by context budget ...]\n"

            snippet, inc = format_chunk(rule, key, clipped)
            ch_len = len(snippet.encode("utf-8", errors="replace"))
            if consumed + ch_len > self.budget_chars:
                return False

            chunks.append(snippet)
            consumed += ch_len
            manifest.append({**inc.__dict__, "budget_after": consumed})
            return True

        roots = scope_roots_anchor(proj, anchor_fp, self.scope_parent_levels)
        anchor_file_s = str(anchor_fp)

        fa = analyzed_files.get(anchor_file_s) or analyzed_files.get(str(anchor_file))
        funcs_here: Dict[str, Dict[str, Any]] = (fa.get("functions") if fa else {}) or {}

        idx = (
            build_symbol_index_for_scope(
                python_files, roots, proj, analyzed_files
            )
            if self.enable_rag or self.k_hop > 0
            else {}
        )

        # --- Parse target callee names (AST; authoritative for rule 'intra_forward') ---
        callee_names_order: List[str] = []
        callee_set: Set[str] = set()
        try:
            tgt_node = ast.parse(textwrap.dedent(target_source)).body
            fn_node = None
            if len(tgt_node) == 1 and isinstance(tgt_node[0], FuncNode):
                fn_node = tgt_node[0]
            elif len(tgt_node) >= 1:
                inner = tgt_node[-1]
                if isinstance(inner, FuncNode):
                    fn_node = inner
            if fn_node:
                nm_set = extract_call_names(fn_node)
                # Stable order by first appearance in preorder walk approx = sorted lexical for determinism:
                callee_set.update(nm_set)
                callee_names_order = sorted(nm_set)

            # Prefer dependency analyzer list as tie-order hint — union with AST
            extra = []
            if deps_list:
                for d in deps_list:
                    if isinstance(d, str) and "." in d:
                        extra.append(d.split(".")[-1])
                    elif isinstance(d, str):
                        extra.append(d)
                callee_set.update(extra)
                callee_names_order = sorted(callee_set)
        except SyntaxError:
            callee_names_order = sorted({d.split(".")[-1] for d in deps_list if isinstance(d, str)})
            callee_set = set(callee_names_order)

        # --- (1) same-file forward callees (sorted) ---
        for nm in callee_names_order:
            hit = find_function_matching_name(funcs_here, nm, exclude_target)
            if not hit:
                continue
            qk, fi = hit
            if qk.split(".")[0] in {"unittest", "sys", "re", "os", "typing"}:
                continue
            blob = fi.get("source") or ""
            add("same_module_callee", f"{qk} @{anchor_fp.name}", blob)

        # --- (2) class-relative envelope ---
        parent_class = None
        if "." in qualified_key:
            parts = qualified_key.split(".")
            if len(parts) >= 2:
                parent_class = parts[0]

        class_budget_soft = max(4096, int(self.budget_chars * self.class_overhead_budget_frac))
        legacy_consumed_before_class = consumed
        if fa and parent_class:
            cls_block = fa.get("classes", {}).get(parent_class)
            if cls_block:
                init_src = cls_block.get("methods", {}).get("__init__", {}).get("source") or ""

                stubs: List[str] = []
                for mname in sorted(cls_block.get("methods", {}).keys()):
                    if mname == qualified_key.split(".")[-1]:
                        continue  # omit target METHOD body duplication
                    minfo = cls_block["methods"][mname]
                    src = minfo.get("source") or ""
                    first_lines = src.splitlines()[:3]
                    header = (
                        "\n".join(first_lines[:2])
                        if first_lines
                        else f"# def {mname}(self, ..."
                    )
                    stubs.append(f"# — method `{mname}` (header only):\n{textwrap.indent(header, ' ' * 4)}")

                class_blob = []
                bases = cls_block.get("bases") or []
                class_blob.append(f"class {parent_class}({', '.join(bases) if bases else ''}):")
                if cls_block.get("docstring"):
                    ds = cls_block["docstring"].strip()
                    class_blob.append(f'    """{ds[:2000]}"""')
                class_blob.append("    # sibling method headers (budgeted excerpt; infer behavior from stubs below):\n")

                sibling_text = "\n".join(stubs[:40])
                if init_src.strip():
                    class_blob.append("    # __init__ excerpt (budgeted):\n")
                    clipped_init = (
                        init_src.strip()
                        if len(init_src) <= class_budget_soft // 3
                        else init_src.strip()[: class_budget_soft // 3]
                        + "\n# [... __init__ truncated ...]\n"
                    )
                    class_blob.append(textwrap.indent(clipped_init, " " * 4))

                class_blob.append(sibling_text)
                class_piece = "\n".join(class_blob)
                reserve = max(0, min(class_budget_soft, self.budget_chars - consumed))
                if reserve > 0 and len(class_piece.encode("utf-8")) > reserve:
                    class_piece = class_piece[:reserve] + "\n# [... class excerpt truncated ...]\n"
                spend_before = consumed
                if add("class_envelope", f"{parent_class} @ {anchor_fp.name}", class_piece):
                    manifest[-1]["class_soft_cap_chars"] = class_budget_soft
                else:
                    consumed = legacy_consumed_before_class

        # --- (3) k-hop frontier (Breadth-first, deterministic) ---
        seen_keys: Set[Tuple[str, str]] = set()

        frontier: deque[Tuple[str, str, int]] = deque()

        unresolved = [
            nm
            for nm in callee_names_order
            if nm not in {qualified_key.split(".")[-1]}
            and find_function_matching_name(funcs_here, nm, exclude_target)
            is None
        ]

        content_cache: Dict[str, str] = {}
        try:
            content_cache[str(anchor_fp)] = anchor_fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass

        for nm in sorted(set(unresolved)):
            frontier.append((nm, anchor_file_s, 0))

        while frontier and consumed < self.budget_chars:
            nm, seed_file, depth = frontier.popleft()
            if depth > self.k_hop:
                continue
            lower = nm.lower()

            defs = sorted(
                (t for t in idx.get(lower, []) if path_allowed(roots, Path(t[0]))),
                key=lambda t: (t[0], t[1]),
            )
            defs = defs[: max(4, len(defs))]
            rag_used = False
            if not defs:
                rag_used = self.enable_rag
                defs = []

            imported = False
            for fp_s, qk, _blurb in defs:
                kk = (fp_s, qk)
                if kk in seen_keys:
                    continue

                fi = analyzed_files.get(fp_s, {}).get("functions", {}).get(qk, {})
                src = fi.get("source")
                if not src:
                    fpath_obj = Path(fp_s)
                    if fp_s not in content_cache:
                        try:
                            content_cache[fp_s] = fpath_obj.read_text(encoding="utf-8", errors="replace")
                        except OSError:
                            continue
                    content = content_cache.get(fp_s) or ""
                    try:
                        tree = ast.parse(content)
                        pts = qk.split(".")
                        fd = _find_qualified_node(tree, pts)
                        if fd:
                            seg = ast.get_source_segment(content, fd)
                            src = seg or ""
                    except SyntaxError:
                        src = ""

                if not src.strip():
                    continue

                imported = True
                seen_keys.add(kk)

                lbl = (
                    "k_hop_forward"
                    if depth > 0
                    else ("scoped_forward" + ("+rag_hint" if rag_used else ""))
                )

                ok = add(lbl, f"{qk} @ {Path(fp_s).name}", src)
                if not ok:
                    continue

                # Expand next hop callee names shallowly inside this snippet
                if depth + 1 <= self.k_hop:
                    try:
                        st = ast.parse(textwrap.dedent(src))
                        body = st.body
                        fb = None
                        if (
                            len(body) == 1
                            and isinstance(body[0], FuncNode)
                        ):
                            fb = body[0]
                        elif body:
                            for n in reversed(body):
                                if isinstance(n, FuncNode):
                                    fb = n
                                    break
                        if fb:
                            for sub in sorted(extract_call_names(fb)):
                                if len(sub) < 2:
                                    continue
                                if sub in _SKIP_BUILTIN_CALL_NAMES:
                                    continue
                                if sub.startswith("__"):
                                    continue

                                frontier.append((sub, fp_s, depth + 1))

                    except SyntaxError:
                        pass

                break

            if imported:
                continue

        header = (
            "# BOUNDED DEPENDENCY CONTEXT\n"
            f"# policy: callee→class→scoped_khop({self.k_hop}) rag={self.enable_rag}; "
            f"budget_chars≈{self.budget_chars}; scope_parent_levels={self.scope_parent_levels}\n\n"
        )
        blob = header + ("\n".join(chunks) if chunks else "# (empty — target only; increase budget/hop)")
        full_text = blob

        return PackedBundle(full_text, manifest)


def attach_bounded_context_to_functions(
    *,
    project_path: str,
    analyzed_files: Dict[str, Any],
    all_functions: Dict[str, Dict[str, Any]],
    python_files: List[str],
    config: Dict[str, Any],
) -> None:
    """Mutate entries in ``all_functions`` with ``context_bundle_text`` / ``context_bundle_manifest``."""
    bundler = BoundedContextBundler(config)

    if not config.get("enable_context_bundle", True):
        for fi in all_functions.values():
            fi["context_bundle_text"] = ""
            fi["context_bundle_raw"] = ""
            fi["context_bundle_manifest"] = []
            fi.pop("context_bundle_policy", None)
        return

    for _fid, fi in all_functions.items():
        fp = fi.get("file_path", "")
        qk = fi.get("qualified_key", "")
        src = fi.get("source_code", "")
        if not fp or not qk:
            fi["context_bundle_text"] = ""
            fi["context_bundle_raw"] = ""
            fi["context_bundle_manifest"] = []
            continue
        pack = bundler.pack(
            project_root=project_path,
            anchor_file=str(fp),
            qualified_key=qk,
            target_source=src,
            analyzed_files=analyzed_files,
            python_files=python_files,
            dependencies=fi.get("dependencies"),
        )

        preview_cap = int(config.get("context_spec_prompt_inject_chars", 16_384))
        full = pack.text.strip()
        fi["context_bundle_raw"] = full
        clipped = truncate_utf8(full, preview_cap)
        if len(clipped.encode("utf-8", errors="replace")) < len(full.encode("utf-8", errors="replace")):
            clipped += (
                "\n# [... context_bundle truncated at "
                "context_spec_prompt_inject_chars (UTF-8 budget) ...]\n"
            )
        fi["context_bundle_text"] = clipped.strip()
        fi["context_bundle_manifest"] = pack.manifest
        fi["context_bundle_policy"] = {
            "budget_chars": bundler.budget_chars,
            "k_hop": bundler.k_hop,
            "scope_parent_levels": bundler.scope_parent_levels,
            "rag_enabled": bundler.enable_rag,
            "manifest_item_count": len(pack.manifest),
        }
