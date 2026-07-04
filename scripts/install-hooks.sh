#!/bin/sh
# Install local git hooks by pointing core.hooksPath at scripts/.
# Run once per clone: ./scripts/install-hooks.sh

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

chmod +x scripts/pre-commit scripts/pre-push
git config core.hooksPath scripts

echo "Git hooks installed (core.hooksPath=scripts)."
echo "  pre-commit: ruff check + ruff format --check"
echo "  pre-push:   pyright + pytest -m \"not integration\" (< 90s target, no containers)"
