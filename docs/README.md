# Documentation Field Guide

Use this guide as the single entry point for all project documentation. It groups every maintained file by intent, shows where to find historical notes, and highlights the fastest command to surface the right doc when you are in a hurry.

## Quick Reference Matrix

| Category | When to Open | Primary Files | Fast Command |
| --- | --- | --- | --- |
| **Project overview** | Need the competition story, deliverables, or high-level architecture. | `docs/project/project-overview.md` | `sed -n '1,80p' docs/project/project-overview.md` |
| **Performance logs** | Reviewing historical runs, regression investigations, or debugging diary notes. | `docs/performance/2025-10-08_convex_hull_debugging.md`, `docs/performance/baseline_2025-10-07*.md`, `docs/performance/baseline_2025-10-08_cache_optimized.md` | `ls docs/performance` |
| **AI handbook** | Need agent operating procedures, post-mortem templates, or debug frameworks. | `docs/ai_handbook/index.md`, `docs/ai_handbook/post_debugging_session_framework.md`, `docs/ai_handbook/debugging_artifacts_organization.md` | `rg --files docs/ai_handbook` |
| **Setup** | Provisioning a new environment or sharing shell helpers. | `docs/setup/SETUP.md`, `docs/setup/setup-uv-env.sh`, `docs/setup/BASH_ALIASES_KO.md` | `ls docs/setup` |
| **Deprecated ideas** | Searching for older proposals that were parked or replaced. | Files under `docs/_deprecated/` | `ls docs/_deprecated` |

## How to Keep This Page Current

1. **Add new docs here immediately.** Include a one-line “When to open” blurb and the preferred quick command.
2. **Retire stale docs.** Move them to `docs/_deprecated/` and update the table to point readers there.
3. **Link from tickets or PRs.** Whenever a discussion references documentation, paste the relevant table row or file path for instant context.

## Search Patterns That Save Time

- List everything at depth ≤ 2 (default morning ritual):
  ```bash
  find docs -maxdepth 2 -type f | sort
  ```
- Jump straight to performance notes from the command line:
  ```bash
  rg "convex" docs/performance
  ```
- Preview the first section of any doc without opening an editor:
  ```bash
  sed -n '1,40p' docs/ai_handbook/index.md
  ```

## Naming Conventions

- Date-stamp detailed logs as `YYYY-MM-DD_topic.md` (`2025-10-08_convex_hull_debugging.md`).
- Keep living references short and topic-focused (`project-overview.md`, `SETUP.md`).
- Archive experiments or abandoned concepts under `_deprecated/`.

Keeping this guide up to date lets every agent skip the guesswork and land on the right documentation in a few keystrokes.
