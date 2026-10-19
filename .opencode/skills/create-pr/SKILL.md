---
name: create-pr
description: Open a GitHub pull request for the current branch once the work is done. Runs the tests, commits any finished work, pushes, and opens a PR against the default branch with a description in a fixed format. Use when the user says the branch is ready, or asks to open, create, or raise a PR.
---

# Create PR

Turn a finished branch into a pull request. Work through the steps in order. If
a step fails, stop and tell the user what failed rather than working around it.

## 1. Check the branch

```bash
base=$(gh repo view --json defaultBranchRef --jq .defaultBranchRef.name)
git fetch origin "$base"
git branch --show-current
git status --short
git log --oneline "origin/$base..HEAD"
```

- If you are on the default branch, stop and ask the user for a branch name.
- If there are no commits ahead and nothing uncommitted, stop: there is nothing
  to open a PR for.

Read the full diff (`git diff "origin/$base...HEAD"`) so the description is
written from what actually changed, not from memory of the session.

## 2. Run the tests

Find the test command in `AGENTS.md`, the README, or the project's build files
(`pyproject.toml`, `package.json`, `Makefile`, `go.mod`, `Cargo.toml`). Run it.

All tests must pass. If any fail, stop and report the failing tests and the
error. Do not skip, delete, or weaken tests to get a pass.

## 3. Commit any remaining work

If there are uncommitted changes that belong to this work, stage them by name
(never `git add -A`) and commit. Do not commit secrets, local config, or files
that `.gitignore` excludes.

```bash
git add <file> ...
git commit -m "<imperative summary, under 72 characters>"
```

## 4. Push

```bash
git push -u origin HEAD
```

## 5. Open the PR, or update the existing one

Write the description to a file so the formatting survives the shell:

```bash
cat > /tmp/pr-body.md <<'EOF'
<body>
EOF
```

If `gh pr view --json url` shows the branch already has an open PR, update it
with `gh pr edit --title "<title>" --body-file /tmp/pr-body.md`. Otherwise:

```bash
gh pr create --base "$base" --title "<title>" --body-file /tmp/pr-body.md
```

Add `--draft` only if the user asked for a draft.

### Title

Imperative mood, under 72 characters. Say what the code now does, not the
symptom the user saw.

### Body

Use exactly these sections, in this order. Leave out a section that genuinely
does not apply rather than writing "N/A".

```markdown
## Summary

One or two sentences: what was wrong or missing, and what this PR does about it.

## Changes

- `path/to/file`: what changed and why.

## Testing

- The test command you ran and its result line, copied verbatim.
- Any new tests and what they guard against.
- Any manual check you ran, with real output. Never invent numbers.

Closes #<issue>
```

### GitHub reference rules

GitHub turns `#<number>` into a link and posts a back-reference on the target.
Editing the text later does not remove it. So:

- Only add `Closes #<n>` for a real issue. Confirm it first with
  `gh issue view <n> --json title,state` and check it matches the work.
  Otherwise leave the line out.
- Do not use `#` for anything else, such as list items or counts.
- Do not write `@name` unless you mean to notify that person.

## 6. Report back

Give the user the PR URL, the test result, and anything you skipped and why.
