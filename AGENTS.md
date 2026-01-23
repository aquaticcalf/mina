This repo is driven by the spec docs in `spec/`. When asked to do any task from `spec/tasks.md` (e.g. “do task 1.1”), follow this workflow every time.

## Workflow

1. **Identify the requested task**
   - Locate the exact task entry in `spec/tasks.md` (e.g. `1.1`, `2.4`, etc.).
   - Treat the task text and its referenced requirement/design notes as the source of truth for scope.

2. **Read the spec in the required order (always)**
   - Read `spec/requirements.md` to understand the functional goals + acceptance criteria relevant to the task.
   - Read `spec/design.md` to understand the intended architecture, types, services, testing strategy, and any constraints relevant to the task.
   - Re-read the specific section(s) of `spec/tasks.md` for the requested item to confirm dependencies/ordering.

3. **Confirm scope and dependencies before editing**
   - Check whether the task depends on earlier tasks (same section/numbering).
   - If prerequisites aren’t done, either implement them first (if clearly required) or ask the user whether to proceed out of order.

4. **Implement exactly what the task asks (no half measures)**
   - Make the minimum code changes needed to satisfy the task end-to-end.
   - Do not leave placeholder work behind (no `TODO:` comments, stubs, or "WIP" hacks).
   - If something blocks completion, stop and ask the user for a decision instead of shipping partials.
   - Keep changes consistent with existing project structure and patterns.
   - Do not “expand scope” beyond what the task and its referenced spec sections require.

5. **Add/Update tests when the task calls for it**
   - If the task includes a property test or unit test step, implement it as part of completing the task.
   - Prefer the test organization and conventions described in `spec/design.md`.

6. **Validate and report**
   - Run the most relevant tests/build checks for the changed area.
   - Report what was changed, where, and how it maps back to the requested task ID.

## How to interpret user requests

- If the user says: “do task `X.Y`”, use `spec/tasks.md` as the execution plan, and use `spec/requirements.md` + `spec/design.md` as constraints and acceptance criteria.
- If the user request conflicts with the spec docs, pause and ask for clarification before proceeding.

## Committing Changes

This repo uses husky with a commitizen hook that requires TTY interaction. Since coding agents don't have TTY access, use the provided commit helper:

```bash
bun commit-agent <type> <subject> [body]
```

### Available Commit Types

- `feat`: A new feature
- `fix`: A bug fix  
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools and libraries such as documentation generation
- `ci`: Changes to our CI configuration files and scripts
- `build`: Changes that affect the build system or external dependencies
- `revert`: Reverts a previous commit

### Examples

```bash
# Implement task from spec/tasks.md
bun commit-agent feat "implement task 1.5 fish disease detection" "Add TensorFlow Lite model integration in app/lib/model/inference.ts"

# Bug fix in model manager
bun commit-agent fix "resolve model loading timeout in manager.ts" "Increased timeout from 5s to 10s for larger TFLite models"

# Add new UI component
bun commit-agent feat "add detection results card component" "Created app/components/ui/detection-card.tsx for displaying disease detection results"

# Test additions
bun commit-agent test "add inference model property tests" "Added hypothesis tests for TFLite model validation in tests/models/inference.test.ts"

# Documentation update
bun commit-agent docs "update spec with new API endpoints" "Added REST API documentation to spec/design.md for external model serving"
```
