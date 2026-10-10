#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
# SPDX-FileCopyrightText: 2026 OGL authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------
# This script triggers a LRZ GitLab CI pipeline on TUM COMA cluster for a specified project and branch.
# Optionally, extra variables can be passed in the form: "variables[KEY]=VALUE".
#----------------------------------------------------------------------------------------
set -euo pipefail

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
if [ $# -lt 4 ]; then
  echo "Usage: $0 <project> <branch> <check_token> <trigger_token> [optional variables]"
  exit 1
fi

PROJECT=$1
BRANCH=$2
CHECK_TOKEN=$3     # read_repository scope
TRIGGER_TOKEN=$4   # LRZ GitLab trigger token
shift 4

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
: "${LRZ_HOST:?Need to set LRZ_HOST}"
: "${LRZ_GROUP:?Need to set LRZ_GROUP}"

# URL-encode branch name
BRANCH_ENC=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$BRANCH")

# -----------------------------------------------------------------------------
# The branch has just been pushed to LRZ GitLab, so it has to exist
# -----------------------------------------------------------------------------
branch_exists=$(curl -s --header "PRIVATE-TOKEN: $CHECK_TOKEN" \
  "https://${LRZ_HOST}/api/v4/projects/${LRZ_GROUP}%2F${PROJECT}/repository/branches/${BRANCH_ENC}" \
  | jq -r '.name // empty')

if [ -z "$branch_exists" ]; then
  echo -e "\033[31m Error: Branch '$BRANCH' does not exist in $PROJECT on LRZ GitLab. Exiting workflow.\033[0m"
  exit 1
fi

# -----------------------------------------------------------------------------
# Build form data
# -----------------------------------------------------------------------------
form_data=(--form "ref=$BRANCH" --form "token=$TRIGGER_TOKEN")
for var in "$@"; do
  form_data+=(--form "$var")
done

# -----------------------------------------------------------------------------
# Trigger pipeline
# -----------------------------------------------------------------------------
echo "Triggering pipeline for project '$PROJECT' on branch '$BRANCH'..."
response=$(curl -s --request POST "${form_data[@]}" \
  "https://${LRZ_HOST}/api/v4/projects/${LRZ_GROUP}%2F${PROJECT}/trigger/pipeline")

pipeline_id=$(echo "$response" | jq -r '.id')

if [ -z "$pipeline_id" ] || [ "$pipeline_id" = "null" ]; then
  echo -e "\033[31m Failed to trigger pipeline for project '$PROJECT' on branch '$BRANCH'.\033[0m"
  echo "$response"
  exit 1
fi

echo "Triggered pipeline $pipeline_id on branch '$BRANCH'."
# Set GitHub Actions output
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "pipeline_id=$pipeline_id" >> "$GITHUB_OUTPUT"
fi
