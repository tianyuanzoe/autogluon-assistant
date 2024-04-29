#!/bin/bash

set -euxo pipefail

LINT_TARGETS="src tests"

black --check $LINT_TARGETS
isort --check-only $LINT_TARGETS
ruff check $LINT_TARGETS

echo "Done! 🥳"
