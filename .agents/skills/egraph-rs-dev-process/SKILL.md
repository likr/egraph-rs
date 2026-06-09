---
name: egraph-rs-dev-process
description: Guidelines for general development process, coding style, writing comments, and updating skills after tasks in the egraph-rs project.
---

# Development Process Guidelines for egraph-rs

This skill details the general development rules, comment conventions, task workflow, and self-updating skill process for the `egraph-rs` project.

## Code Comment Rules
- **Language**: All comments in code files (Rust, Python, JS, TS) must be written in English.
- **Content**: Write comments that explain the **WHY**, not the **HOW**. Explain design decisions, algorithms, and logical justifications.
- **Cleanliness**: Avoid unnecessary commented-out code or work-in-progress notes.

## Task Execution Workflow

### 1. Before Starting a Task
- Understand task instructions and constraints.
- Refer to `egraph-rs-system-patterns` skill to understand the project architecture.
- Examine related code files to understand existing implementation patterns.
- Create an internal task checklist (e.g. `task.md`) to plan and track your work.

### 2. During Implementation
- Ensure all comments adhere to the "Code Comment Rules" above.
- Follow formatting and style guidelines.
- Regularly verify changes.

### 3. Before Task Completion (Final Verification Process)
- **Formatting and Linting**:
  - Run `cargo fmt --all` and ensure no formatting changes remain.
  - Run `cargo clippy --workspace --all-targets --all-features -- -D warnings` and ensure there are no warnings.
- **Run Tests**:
  - Verify that the changes do not break existing code by running workspace tests.
  - See `egraph-rs-verification` skill for specific test commands.
- **Prepare Commit**:
  - Draft a commit message following Conventional Commits format (see `egraph-rs-commit` skill).
- **Ask for User Confirmation**:
  - Present a summary of all changes made.
  - Include the proposed commit message.
  - Wait for explicit user approval before completing the task. **This confirmation step must never be skipped under any circumstances.**

## Post-Task Reflection: Updating Skills
Upon completing any task, the agent **MUST** reflect on the work performed.
- Analyze if new implementation patterns, codebase quirks, build command requirements, or general instructions were learned during the task.
- If new patterns or insights emerged, immediately update the relevant skill files (e.g., `egraph-rs-dev-process`, `egraph-rs-verification`, `egraph-rs-system-patterns`) to keep the guidelines up-to-date.
- This self-updating loop replaces the legacy "Memory Bank" mechanism and ensures future agents benefit from the context established in this task.
