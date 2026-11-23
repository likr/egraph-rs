---
name: egraph-rs-github
description: Git commit, push, branch/tag immutability, Pull Request, and Issue workflow conventions for egraph-rs. Use whenever committing, pushing, managing branches/tags, or opening/reviewing GitHub Issues and PRs.
---

# Git & GitHub Workflow Skill for `egraph-rs`

Follow these conventions alongside [`.agents/AGENTS.md`](../../AGENTS.md), [`egraph-rs-commit`](../egraph-rs-commit/SKILL.md), [`egraph-rs-release`](../egraph-rs-release/SKILL.md), and [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

## 1. Branch & Worktree Discipline

- **`master` Integration**: Keep `master` linear, buildable, and verified (`./scripts/verify.sh`).
- **Topic Branches**: Use Conventional Commit prefixes (`feat/<slug>`, `fix/<slug>`, `perf/<slug>`, `docs/<slug>`, `chore/<slug>`).
- **Parallel Worktrees**: Use [`scripts/worktree.sh`](../../../scripts/worktree.sh) (`wt/<slug>` under `.worktrees/<slug>`) so each concurrent author has isolated `target/` and `.venv/`. Never push temporary `wt/*` branches to `origin`.

## 2. Commit & Push Rules

1. **Pre-Commit Hygiene**:
   - Inspect `git status -sb` and `git --no-pager diff --cached --stat` before committing.
   - Split unrelated edits (e.g., library code vs. `.github/` templates vs. `.agents/` updates) into separate atomic commits.
   - Verify no secrets, credentials, gitignored outputs (`target/`, `.venv/`, `Cargo.lock`, `crates/wasm/pkg/`, `js/examples/dist/`), non-English code/doc/commit text (`jp_cjk=0`), or files `> 100 MB` are staged.
2. **Pre-Commit Confirmation Gate**:
   - By default, present the Japanese summary and proposed Conventional Commit message and obtain explicit user approval before committing or pushing (`egraph-rs-dev-process`). Only commit/push autonomously when the user explicitly authorized autonomous commits/pushes for the active run.
3. **Pre-Push Verification**:
   - Run [`scripts/verify.sh`](../../../scripts/verify.sh) (`--fast`, `--rust`, `--python`, `--wasm`, `--invariants`, `--docs`, or `--all --build-wasm`) matching the scope of changes before pushing `origin/master`.
   - Confirm `git rev-parse HEAD origin/master` alignment after pushing and verify both `/Users/likr/egraph-rs` and any paired repos remain clean.
4. **History & Tag Immutability**:
   - **Never** run `git push --force` / `--force-with-lease` on `master`.
   - **Never** move or delete pushed release tags (`v6.*`). Recover from a failed release workflow by fixing `master` and tagging the next version (`egraph-rs-release`).

## 3. GitHub Issues & Pull Requests (`gh` CLI)

- **Language**: Write all GitHub Issue titles/bodies/comments, PR descriptions, and release notes strictly in English (`jp_cjk=0`).
- **Templates & Labels**:
  - Issues must align with `.github/ISSUE_TEMPLATE/` (`bug_report.yml`, `feature_request.yml`, `release_checklist.md`) and include reproducible parameters + explicit `u64` seeds when reporting bugs or regressions.
  - PRs must complete `.github/pull_request_template.md`, link closing issues (`Closes #<num>`), state verified single-thread commands, and keep diff scope minimal.
  - Standard labels: `bug`, `enhancement`, `triage`, `release`, `documentation`, `ci`.
- **Non-Polling CI Checks**:
  - When waiting for `.github/workflows/ci.yml` or `release.yml` via `gh`, resolve the triggered run ID (`gh run list --workflow ... --limit 3 --json databaseId,headSha,status,url`) and monitor once in the background with `gh run watch <id> --exit-status` instead of `sleep` loops.
  - Ensure all 6 `CI` jobs pass before merging PRs or pushing release tags.
