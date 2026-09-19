#!/usr/bin/env python3
"""Static read/write column-contract check across the PhaseFormer-L stages.

Why this exists
---------------
A real defect was found in this suite by hand: `e16_writeback.py` read a column
named ``majority_input_group_label`` while the producer
(`e16_dissection.py`) writes ``leading_input_group_label``.  The write-back
would have silently blanked a section 4.4 column *after* the expensive
dissection run had already finished.

That is a whole *class* of defect -- "the consumer names a column the producer
never writes" -- and hand-checking does not scale.  This script detects the
class statically:

* it derives each producer's column set from the producer's **own source**
  (the dict literals that build the rows, or the module's declared
  ``RESULTS_FIELDS``/``TABLE_FIELDS``), so it cannot drift from the code;
* it derives each consumer's column set from the names the consumer actually
  subscripts (``row["x"]``) or reads (``row.get("x")``);
* it reports every consumer column that no producer for that input supplies.

Producers whose schema is dynamic (built from a runtime dict) are expanded with
the band-name templates read out of ``band_summary``.

Scope and honesty
-----------------
This checks **names only**.  It cannot show that two columns share a meaning, a
unit, or a join key, and a column that is present but always empty still passes
here -- that is what ``check_builder_outputs.py`` (empty-column sweep) and the
per-experiment stage-5 audit are for.

Usage::

    python scripts/phaseformer_L/check_column_contracts.py            # report
    python scripts/phaseformer_L/check_column_contracts.py --strict   # exit 1 on any gap
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# --------------------------------------------------------------------------
# Producer-side extraction
# --------------------------------------------------------------------------

def _dict_keys_in(node: ast.AST) -> set:
    """Literal string keys *defined* under ``node``.

    Two forms both define a row/record column and both occur in this suite:

    * a dict literal -- ``{"dataset": ...}``;
    * a subscript store -- ``record["revin"] = {...}``, which is how the
      projector audit builds its nested blocks (missing this form made the
      checker report a false gap on ``revin``).
    """
    keys: set = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Dict):
            for key in sub.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
        if (isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store)
                and isinstance(sub.slice, ast.Constant)
                and isinstance(sub.slice.value, str)):
            keys.add(sub.slice.value)
    return keys


def _literal_assignment(path: Path, names: set) -> dict:
    """Top-level ``NAME = (...)`` string-tuple constants, by name."""
    tree = ast.parse(path.read_text())
    out: dict = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in names:
                    try:
                        out[target.id] = set(ast.literal_eval(node.value))
                    except Exception:
                        out[target.id] = set()
    return out


def _function_dict_keys(path: Path, func_names: set) -> set:
    """Dict-literal keys inside the named top-level functions."""
    tree = ast.parse(path.read_text())
    keys: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            keys |= _dict_keys_in(node)
    return keys


def _enclosing_function_keys(path: Path, marker: str) -> set:
    """Dict-literal keys inside EVERY function mentioning ``marker``.

    Union across all occurrences, not just the first: a record's schema is often
    assembled in one function and serialised in another, and anchoring on the
    first hit would miss the half that defines the keys.
    """
    source = path.read_text()
    tree = ast.parse(source)
    hit_lines = [i for i, line in enumerate(source.splitlines(), start=1)
                 if marker in line]
    if not hit_lines:
        return set()
    keys: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            end = node.end_lineno or node.lineno
            if any(node.lineno <= line <= end for line in hit_lines):
                keys |= _dict_keys_in(node)
    return keys


def _band_summary_keys() -> set:
    """Expand ``band_summary``'s f-string templates with E16's band names.

    Both sides are read from source: the suffixes come from the f-strings in
    ``evaluate_lowrank_semantic_interventions.band_summary``, the band names
    from the ``bands`` dict literal in ``e16_dissection.build_bases``.
    """
    band_mod = REPO_ROOT / "scripts" / "evaluate_lowrank_semantic_interventions.py"
    tree = ast.parse(band_mod.read_text())
    suffixes: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "band_summary":
            for sub in ast.walk(node):
                if isinstance(sub, ast.JoinedStr):
                    parts = []
                    for piece in sub.values:
                        if isinstance(piece, ast.Constant):
                            parts.append(str(piece.value))
                        else:
                            parts.append("{name}")
                    text = "".join(parts)
                    if "{name}" in text:
                        suffixes.add(text)
    e16 = REPO_ROOT / "scripts" / "phaseformer_L" / "e16_dissection.py"
    band_names: set = set()
    for node in ast.walk(ast.parse(e16.read_text())):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if "random" in keys and "dimension" in keys and "repeats" in keys:
                band_names |= {k for k in keys if "dimension" not in k and "repeats" not in k
                               and "family" not in k and "seed" not in k and "bases" not in k}
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and "rrr" in str(target.slice.value)):
                    band_names.add(str(target.slice.value))
    expanded: set = set()
    for suffix in suffixes:
        for name in band_names:
            expanded.add(suffix.replace("{name}", name))
    return expanded


def producer_schemas() -> dict:
    e16 = REPO_ROOT / "scripts" / "phaseformer_L" / "e16_dissection.py"
    e17 = REPO_ROOT / "scripts" / "phaseformer_L" / "e17_conditional.py"
    e18 = REPO_ROOT / "scripts" / "phaseformer_L" / "e18_negative.py"
    e14t = REPO_ROOT / "scripts" / "phaseformer_L" / "e14_read_test.py"
    svd = REPO_ROOT / "scripts" / "phaseformer_L" / "e18_svd_truncation.py"
    generic = REPO_ROOT / "scripts" / "phaseformer_L" / "read_test_generic.py"

    generic_cols = _literal_assignment(generic, {"TEST_COLUMNS"}).get("TEST_COLUMNS", set())

    intervention = _enclosing_function_keys(e16, "intervention_rows.append") | _band_summary_keys()
    dissection = _function_dict_keys(e16, {"dissection_rows"}) | _band_summary_keys()

    return {
        "e16_intervention_table.csv": intervention,
        "e16_dissection_table.csv": dissection,
        "e17_results.with_test.csv": (
            _literal_assignment(e17, {"RESULTS_FIELDS"}).get("RESULTS_FIELDS", set()) | generic_cols
        ),
        "e18_results.with_test.csv": (
            _literal_assignment(e18, {"RESULTS_FIELDS"}).get("RESULTS_FIELDS", set()) | generic_cols
        ),
        "e14_results.csv": _literal_assignment(e14t, {"RESULTS_FIELDS"}).get("RESULTS_FIELDS", set()),
        "e18_svd_truncation_table_28.csv": (
            _literal_assignment(svd, {"TABLE_FIELDS"}).get("TABLE_FIELDS", set())
        ),
        # The frozen projector audit (JSON).  Its producer is a *frozen* stage
        # script that the pipeline deliberately does not invoke, so the contract
        # is implicit -- worth checking precisely because it is implicit.
        "e17_projector_audit.json": _enclosing_function_keys(
            REPO_ROOT / "scripts" / "phaseformer_L" / "e17_conditional_projectors.py",
            '"projector_audit.json"',
        ),
    }


# --------------------------------------------------------------------------
# Consumer-side extraction
# --------------------------------------------------------------------------

def consumer_keys(path: Path) -> tuple:
    """(columns READ, columns WRITTEN, f-string WRITE patterns).

    Only reads matter for a read/write contract, and the distinction has to be
    syntactic or the check drowns in false positives: a write-back builds its
    *output* rows as dict literals (``{"dataset": ..., "verdict": ...}``) and
    those keys are written, never demanded of the producer.  So:

    * reads  -- ``row["x"]`` in a load context, and ``row.get("x")``;
    * writes -- keys of any dict literal, and ``row["x"] = ...`` (store context).
    """
    tree = ast.parse(path.read_text())
    reads: set = set()
    writes: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                if isinstance(node.ctx, ast.Store):
                    writes.add(node.slice.value)
                else:
                    reads.add(node.slice.value)
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    writes.add(key.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"get", "pop"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    reads.add(first.value)
    # Keys can also be built by f-string ("entry[f\"{arm}_mse\"]"), which a
    # literal scan cannot see.  Collect those as patterns so a later literal read
    # of "direct_mse" is recognised as internally constructed rather than a
    # demand on the producer.  Without this the check cries wolf on every
    # f-string-built pivot column, which is how this report earns being ignored.
    templates: set = set()
    for node in ast.walk(tree):
        candidates = []
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            candidates.append(node.slice)
        if isinstance(node, ast.Dict):
            candidates.extend(node.keys)
        for cand in candidates:
            if isinstance(cand, ast.JoinedStr):
                parts = []
                for piece in cand.values:
                    if isinstance(piece, ast.Constant):
                        parts.append(re.escape(str(piece.value)))
                    else:
                        parts.append("[A-Za-z0-9_]+")
                templates.add("".join(parts))

    clean = lambda s: {k for k in s if k and not k.startswith("_")}
    return clean(reads), clean(writes), templates


#: consumer script -> which producer schemas together form its input columns
CONTRACTS = {
    "e16_writeback.py": ["e16_intervention_table.csv", "e16_dissection_table.csv"],
    "e17_writeback.py": ["e17_results.with_test.csv", "e17_projector_audit.json"],
    "e18_writeback.py": ["e18_results.with_test.csv", "e14_results.csv",
                         "e18_svd_truncation_table_28.csv"],
}

#: Tokens that are dict/config keys rather than CSV columns; reported separately
#: so a genuine gap is not hidden among them.
NON_COLUMN_HINTS = {
    "event", "status", "reason", "path", "value", "name", "seed", "kind",
    "label", "note", "source", "detail", "root", "key", "meta", "error",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when any consumer column has no producer")
    parser.add_argument("--output", default="",
                        help="write the full report as JSON to this path")
    args = parser.parse_args()

    producers = producer_schemas()
    report: dict = {"producers": {k: sorted(v) for k, v in producers.items()}, "contracts": {}}
    gaps_total = 0

    print("=== producer column sets (from source) ===")
    for name, cols in producers.items():
        print(f"  {name:36s} {len(cols):3d} columns")

    print("\n=== consumer -> producer reconciliation ===")
    for consumer, inputs in CONTRACTS.items():
        path = HERE / consumer
        if not path.is_file():
            print(f"  !! {consumer} not found")
            continue
        supplied: set = set()
        for schema in inputs:
            supplied |= producers.get(schema, set())
        reads, writes, templates = consumer_keys(path)
        compiled = [re.compile(rf"^(?:{pat})$") for pat in templates]
        internally_built = {k for k in reads if any(rx.match(k) for rx in compiled)}
        # read, never written literally, never built by an f-string template
        demanded = reads - writes - internally_built
        missing = sorted(k for k in demanded - supplied if k not in NON_COLUMN_HINTS)
        non_column = sorted(k for k in demanded - supplied if k in NON_COLUMN_HINTS)
        report["contracts"][consumer] = {
            "inputs": inputs,
            "columns_read": len(reads),
            "columns_written": len(writes),
            "input_columns_demanded": len(demanded),
            "internally_built_keys": sorted(internally_built),
            "missing_from_producers": missing,
            "unclassified_tokens": non_column,
        }
        gaps_total += len(missing)
        flag = "OK " if not missing else "GAP"
        print(f"  {flag} {consumer:22s} read={len(reads):3d} written={len(writes):3d} "
              f"input-demanded={len(demanded):3d} missing={len(missing)}")
        for key in missing:
            print(f"        MISSING: {key}")
        if non_column:
            print(f"        (non-column tokens, not treated as gaps: {non_column})")

    print(f"\ntotal missing consumer columns: {gaps_total}")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"report written to {args.output}")
    if args.strict and gaps_total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
