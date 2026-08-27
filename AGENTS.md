# Codex Repository Instructions

Before working in this repository, read and follow `MANAGE_RULES.md`. For
research tasks, also follow `HOW_TO_DO_RESEARCH.md` and the active experiment
plan referenced there.

When the user provides a concrete model hypothesis and asks Codex to implement
it, run experiments, and analyze high-error or significantly degraded samples,
use `$experiment-and-error-analysis` from
`.agents/skills/experiment-and-error-analysis/`.

Do not trigger that skill for hypothesis discussion alone, analysis of an
already supplied aggregate result, smoke tests alone, or requests that
explicitly exclude experiment execution.

## GitHub pull/push network route

Use GitHub's dedicated SSH-over-443 endpoint through the local SOCKS5 proxy
on port 7897. The proxy target must be `ssh.github.com:443` (using
`github.com:443` can hang during the SSH banner exchange).

For pull/fetch:

```bash
git -c core.sshCommand='ssh -o HostName=ssh.github.com -o ConnectTimeout=15 -o ProxyCommand="nc -x 127.0.0.1:7897 -X 5 ssh.github.com 443" -p 443' pull
```

For push, use the same temporary SSH command:

```bash
git -c core.sshCommand='ssh -o HostName=ssh.github.com -o ConnectTimeout=15 -o ProxyCommand="nc -x 127.0.0.1:7897 -X 5 ssh.github.com 443" -p 443' push
```

This leaves the repository's remote URL unchanged. If local and remote
branches have diverged, omit `--ff-only` only when preserving local commits
and merging is intended.
