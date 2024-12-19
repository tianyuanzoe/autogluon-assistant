#!/bin/bash
set -ex

source $(dirname "$0")/env_setup.sh

install_coverage_test

COVERAGE_FILE="./ui_coverage.json"

coverage-threshold --line-coverage-min 80 --file-line-coverage-min 80 --coverage-json "$COVERAGE_FILE"
