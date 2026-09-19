#!/usr/bin/env python3
"""Static check of the phase-2 pipeline's invocations against the target scripts.

Why this exists
---------------
The phase-2 chain is unattended and long: it runs E14's test read, then E16, E17
and E18, whose training alone is 165 runs.  A mistake in a single argument is
therefore only discovered *late*, and by then the expensive stages are already
spent.

Two real defects of this family were found by hand in this suite:

1. step 6 passed ``--e14-results .../results.with_test.csv`` -- a file E14 never
   produces (it fills ``results.csv`` in place).  Caught by checking that each
   argument's VALUE resolves to something a producer writes.
2. step 6 passed ``--seeds 2021 2022 2023``.  ``--seeds`` is parsed with
   ``parse_list`` (comma-separated) and is NOT declared ``nargs=...``, so
   argparse consumed ``2021`` and rejected the rest::

       error: unrecognized arguments: 2022 2023        (exit 2)

   An earlier audit had verified that every flag *exists* in the target's
   ``add_argument`` list -- which is exactly why it missed this.  **Flag
   existence is not flag arity.**

What it checks
--------------
* every ``--flag`` used in the pipeline is declared by the target script;
* a flag followed by two or more bare values must be declared with ``nargs``
  (otherwise the extra tokens are argparse errors).  Flags consumed as
  comma-separated lists are expected to be given as ONE token.

Limits: it cannot know whether a *single* value is semantically right (that is
the naming/column contract checks' job), and it does not run the commands.

Usage::

    python scripts/phaseformer_L/check_pipeline_invocations.py [--pipeline PATH]
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_PIPELINE = HERE / "run_phase2_after_e14.sh"

INVOCATION = re.compile(r"\$PY'?\s+(scripts/[\w/]+\.py)([^&;|]*)")


def declared_flags(script: pathlib.Path) -> dict:
    """flag -> has_nargs, from the target script's argparse."""
    if not script.is_file():
        return {}
    tree = ast.parse(script.read_text())
    out: dict = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        flag = None
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and arg.value.startswith("--"):
                flag = arg.value
                break
        if flag is None:
            continue
        has_nargs = any(kw.arg == "nargs" for kw in node.keywords)
        out[flag] = has_nargs
    return out


def strip_quotes(token: str) -> str:
    return token.strip().strip("'\"")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", default=str(DEFAULT_PIPELINE))
    args = parser.parse_args()

    text = pathlib.Path(args.pipeline).read_text()
    # join backslash continuations, and drop the trailing prose comment block
    body = text.split("# NOTE on --verify semantics")[0].replace("\\\n", " ")

    problems: list = []
    checked = 0
    for match in INVOCATION.finditer(body):
        rel, tail = match.group(1), match.group(2)
        script = REPO_ROOT / rel
        flags = declared_flags(script)
        tokens = [strip_quotes(tok) for tok in tail.split()]

        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not token.startswith("--"):
                index += 1
                continue
            # gather the values belonging to this flag, up to the next flag
            values = []
            look = index + 1
            while look < len(tokens) and not tokens[look].startswith("--"):
                values.append(tokens[look])
                look += 1
            checked += 1
            if token not in flags:
                problems.append(f"{rel}: {token} is not declared by the script")
            elif len(values) >= 2 and not flags[token]:
                problems.append(
                    f"{rel}: {token} got {len(values)} bare values "
                    f"({values}) but is not declared with nargs; argparse would "
                    f"reject the extras. Use a comma-separated single value."
                )
            index = look

    print(f"pipeline: {args.pipeline}")
    print(f"flags inspected: {checked}")
    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("\nOK: every flag is declared, and no non-nargs flag receives multiple values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
