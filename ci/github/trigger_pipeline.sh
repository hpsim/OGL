#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
# -----------------------------------------------------------------------------
# Trigger a new LRZ GitLab CI pipeline on TUM COMA cluster for a given branch.
#
# Uses environment variables defined in the GitHub Actions job:
#   LRZ_GROUP, LRZ_HOST, REPO_NAME, LRZ_GITLAB_TRIGGER_TOKEN
#
# Usage:
#   ./ci/github/trigger_pipeline.sh <branch>
# -----------------------------------------------------------------------------

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <branch>"
  exit 1
fi

GROUP=greole
HOST="gitlab-ce.lrz.de"
PROJECT=$1
BRANCH=$2
CHECK_TOKEN=$3
TRIGGER_TOKEN=$4
shift 4
VARIABLES="$@"     # Optional extra variables in the form: "variables[KEY]=VALUE"
OGL_BRANCH="$BRANCH"


# -----------------------------------------------------------------------------
# Determine ogl branch on LRZ GitLab
# -----------------------------------------------------------------------------
OGL_PROJECT="ogl"

echo "Checking if ogl branch '${BRANCH}' exists on LRZ GitLab..."

# Encode branch name for safe URL use
#ENCODED_BRANCH=$(printf '%s' "$BRANCH" | jq -sRr @uri)
#
## Query the ogl repo branch endpoint on LRZ GitLab
#status_code=$(curl -s -o /dev/null -w "%{http_code}" \
#  "https://${HOST}/api/v4/projects/${GROUP}%2F${OGL_PROJECT}/repository/branches/${ENCODED_BRANCH}")
#
#
#if [ "$status_code" -eq 200 ]; then
#  OGL_BRANCH="$BRANCH"
#  echo "ogl branch '${BRANCH}' exists on LRZ GitLab."
#else
#  OGL_BRANCH="develop"
#  echo "ogl branch '${BRANCH}' does not exist on LRZ GitLab. Using '${OGL_BRANCH}'."
#fi

# -----------------------------------------------------------------------------
# Trigger NeoFOAM pipeline
# -----------------------------------------------------------------------------
echo "Triggering a new CI pipeline on branch $BRANCH in project: $GROUP/$PROJECT"
echo "Use ogl branch: $OGL_BRANCH"

# Prepare curl form data for variables
FORM_DATA="--form ref=$BRANCH --form token=$TRIGGER_TOKEN"
FORM_DATA="$FORM_DATA --form variables[OGL_BRANCH]=$OGL_BRANCH"

for var in $VARIABLES; do
  FORM_DATA="$FORM_DATA --form $var"
done

response=$(curl -s --request POST $FORM_DATA \
  "https://${HOST}/api/v4/projects/${GROUP}%2F${PROJECT}/trigger/pipeline")

echo "$response" | jq .

pipeline_id=$(echo "$response" | jq -r '.id')
if [ "$pipeline_id" = "null" ] || [ -z "$pipeline_id" ]; then
  echo "Failed to trigger LRZ CI pipeline"
  exit 1
fi

echo "Successfully triggered pipeline: $pipeline_id"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "pipeline_id=$pipeline_id" >> "$GITHUB_OUTPUT"
fi
