---
name: egraph-rs-commit
description: Conventions and format rules for writing commit messages in the egraph-rs project.
---

# Commit Message Conventions for egraph-rs

All commit messages for changes in this repository must strictly adhere to the **Conventional Commits** format.

## Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

## Permitted Types
- `feat`: A new feature (e.g. implementing a new layout variant)
- `fix`: A bug fix (e.g. correcting a loop or edge index calculation)
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (formatting, lint repairs, white-space)
- `refactor`: A code change that neither fixes a bug nor adds a feature (e.g. cleaning up generics)
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process, scripts, configuration, or dependency updates

## Scoping Rules
- **Crate-specific changes**: Always specify the crate name as the scope if the modification is localized to one crate.
  - Examples: `egraph-wasm`, `petgraph-layout-mds`, `petgraph-layout-kernel-sgd`
- **Project-wide changes**: Omit the scope for root configuration, documentation, repository guidelines, or broad-sweeping multi-crate modifications.
  - Example: `docs: update skills and remove memory bank`

## Examples
- `feat(petgraph-layout-kernel-sgd): add DiffusionKernel random access interface`
- `fix(petgraph-clustering): correct Louvain modularity calculation`
- `test(petgraph-layout-sgd): add comprehensive tests for schedulers`
- `refactor(petgraph-algorithm-shortest-path): optimize distance calculation`
- `docs: update project workflow guidelines` (broad repo change)
