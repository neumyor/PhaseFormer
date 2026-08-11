# Claude Code Repository Instructions

Before working in this repository, read and follow `MANAGE_RULES.md`. For
research tasks, also follow `HOW_TO_DO_RESEARCH.md` and the active experiment
plan referenced there.

When the user provides a concrete model hypothesis and asks Claude Code to
implement it, run experiments, and analyze high-error or significantly degraded
samples, use `/experiment-and-error-analysis` from
`.claude/skills/experiment-and-error-analysis/`.

Do not trigger that skill for hypothesis discussion alone, analysis of an
already supplied aggregate result, smoke tests alone, or requests that
explicitly exclude experiment execution.
