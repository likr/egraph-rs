#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/worktree.sh <subcommand> [args...]

Subcommands:
  create <slug> [base_ref]             Create .worktrees/<slug> on branch wt/<slug> and setup .venv
  list                                 List all git worktrees, branches, dirty status, and .venv presence
  verify <slug> [verify.sh args...]    Run scripts/verify.sh inside .worktrees/<slug>
  clean <slug> [--delete-branch] [--force]
                                       Remove .worktrees/<slug> safely and optionally delete wt/<slug>
  prune                                Prune stale git worktree administrative metadata
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

MAIN_ROOT="$(git worktree list --porcelain | awk '/^worktree / {print $2; exit}')"
if [[ -z "$MAIN_ROOT" ]]; then
  echo "Error: must be run inside a git repository." >&2
  exit 1
fi

validate_slug() {
  local slug="$1"
  if [[ ! "$slug" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
    echo "Invalid worktree slug '$slug' (expected lowercase [a-z0-9._-])." >&2
    exit 2
  fi
}

CMD="$1"
shift

case "$CMD" in
  create)
    if [[ $# -lt 1 ]]; then
      echo "Usage: scripts/worktree.sh create <slug> [base_ref]" >&2
      exit 2
    fi
    SLUG="$1"
    BASE_REF="${2:-HEAD}"
    validate_slug "$SLUG"

    WT_DIR="$MAIN_ROOT/.worktrees/$SLUG"
    BRANCH="wt/$SLUG"

    if [[ -e "$WT_DIR" ]]; then
      echo "Error: worktree directory already exists: $WT_DIR" >&2
      exit 1
    fi
    if git -C "$MAIN_ROOT" show-ref --verify --quiet "refs/heads/$BRANCH"; then
      echo "Error: branch '$BRANCH' already exists. Clean or pick another slug." >&2
      exit 1
    fi

    mkdir -p "$MAIN_ROOT/.worktrees"
    git -C "$MAIN_ROOT" worktree add -b "$BRANCH" "$WT_DIR" "$BASE_REF"
    if [[ -f "$MAIN_ROOT/Cargo.lock" && ! -f "$WT_DIR/Cargo.lock" ]]; then
      cp "$MAIN_ROOT/Cargo.lock" "$WT_DIR/Cargo.lock"
    fi
    "$MAIN_ROOT/scripts/setup_venv.sh" "$WT_DIR/.venv"

    echo ""
    echo "[worktree] Ready: $WT_DIR (branch: $BRANCH, base: $BASE_REF)"
    echo "Recommended parallel execution env:"
    echo "  export EGRAPH_PARALLEL_WORKERS=2   # or 3-4 depending on active workers"
    echo "  export VIRTUAL_ENV=\"$WT_DIR/.venv\""
    echo "  export RUSTC_WRAPPER=sccache"
    echo "  cd \"$WT_DIR\""
    ;;

  list)
    python3 - "$MAIN_ROOT" <<'PY'
import pathlib, subprocess, sys

main_root = pathlib.Path(sys.argv[1])
out = subprocess.check_output(
    ["git", "-C", str(main_root), "worktree", "list", "--porcelain"],
    text=True,
)
entries = []
cur = {}
for line in out.splitlines():
    if not line.strip():
        if cur:
            entries.append(cur)
            cur = {}
        continue
    k, _, v = line.partition(" ")
    cur[k] = v
if cur:
    entries.append(cur)

print(f"{'PATH':<55} {'BRANCH':<28} {'SHA':<9} {'STATUS':<12} {'VENV'}")
print("-" * 116)
for e in entries:
    path = pathlib.Path(e["worktree"])
    branch = e.get("branch", "(detached)").replace("refs/heads/", "")
    sha = e.get("HEAD", "")[:7]
    if not path.exists():
        status = "missing"
        venv = "no"
    else:
        st = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True,
        ).splitlines()
        status = f"dirty({len(st)})" if st else "clean"
        venv = "yes" if (path / ".venv" / "bin" / "python").exists() else "no"
    print(f"{str(path):<55} {branch:<28} {sha:<9} {status:<12} {venv}")
PY
    ;;

  verify)
    if [[ $# -lt 1 ]]; then
      echo "Usage: scripts/worktree.sh verify <slug> [verify.sh args...]" >&2
      exit 2
    fi
    SLUG="$1"
    shift
    validate_slug "$SLUG"
    WT_DIR="$MAIN_ROOT/.worktrees/$SLUG"
    if [[ ! -d "$WT_DIR" ]]; then
      echo "Error: worktree not found: $WT_DIR" >&2
      exit 1
    fi
    VERIFY_BIN="$WT_DIR/scripts/verify.sh"
    if [[ ! -x "$VERIFY_BIN" ]]; then
      VERIFY_BIN="$MAIN_ROOT/scripts/verify.sh"
    fi
    (cd "$WT_DIR" && exec "$VERIFY_BIN" "$@")
    ;;

  clean)
    if [[ $# -lt 1 ]]; then
      echo "Usage: scripts/worktree.sh clean <slug> [--delete-branch] [--force]" >&2
      exit 2
    fi
    SLUG="$1"
    shift
    validate_slug "$SLUG"
    DELETE_BRANCH=0
    FORCE=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --delete-branch) DELETE_BRANCH=1; shift ;;
        --force) FORCE=1; shift ;;
        *)
          echo "Unknown option for clean: $1" >&2
          exit 2
          ;;
      esac
    done

    WT_DIR="$MAIN_ROOT/.worktrees/$SLUG"
    BRANCH="wt/$SLUG"

    if [[ -d "$WT_DIR" ]]; then
      DIRTY="$(git -C "$WT_DIR" status --porcelain)"
      if [[ -n "$DIRTY" && "$FORCE" -eq 0 ]]; then
        echo "Error: worktree '$SLUG' has uncommitted changes. Commit, stash, or pass --force." >&2
        git -C "$WT_DIR" status -sb >&2
        exit 1
      fi
      if [[ "$FORCE" -eq 1 ]]; then
        git -C "$MAIN_ROOT" worktree remove --force "$WT_DIR"
      else
        git -C "$MAIN_ROOT" worktree remove "$WT_DIR"
      fi
    else
      echo "[worktree] Directory already absent: $WT_DIR (pruning)"
    fi

    git -C "$MAIN_ROOT" worktree prune -v

    if [[ "$DELETE_BRANCH" -eq 1 ]] && git -C "$MAIN_ROOT" show-ref --verify --quiet "refs/heads/$BRANCH"; then
      if [[ "$FORCE" -eq 1 ]]; then
        git -C "$MAIN_ROOT" branch -D "$BRANCH"
      else
        git -C "$MAIN_ROOT" branch -d "$BRANCH"
      fi
    fi
    echo "[worktree] Cleaned '$SLUG'."
    ;;

  prune)
    git -C "$MAIN_ROOT" worktree prune -v
    ;;

  -h|--help)
    usage
    ;;

  *)
    echo "Unknown subcommand: $CMD" >&2
    usage >&2
    exit 2
    ;;
esac
