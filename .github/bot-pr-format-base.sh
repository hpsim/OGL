#!/usr/bin/env bash

source .github/bot-pr-base.sh

echo "Retrieving PR file list"
PR_FILES=$(bot_get_all_changed_files ${PR_URL})
NUM=$(echo "${PR_FILES}" | wc -l)
echo "PR has ${NUM} changed files"

TO_FORMAT="$(echo "$PR_FILES" | grep -E $EXTENSION_REGEX || true)"

git remote add fork "$HEAD_URL"
git fetch fork "$HEAD_BRANCH"

git config user.email "go@hpsim.de"
git config user.name "OGLBot"

# checkout current PR head
LOCAL_BRANCH=format-tmp-$HEAD_BRANCH
git checkout -b $LOCAL_BRANCH fork/$HEAD_BRANCH
