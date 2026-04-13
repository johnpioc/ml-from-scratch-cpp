# CLAUDE.md

Standing instructions for Claude Code when working in this repository.

## Project overview

`jmll` is a from-scratch C++ machine learning library built as a deliberate learning project. The goal is to develop industry-grade C++ skills targeting high-performance roles such as quantitative developer positions. The library is **not** a production tool — it is a skills-building vehicle, and architectural and implementation decisions are made with career relevance in mind.

Currently implemented:
- k-Nearest Neighbours (classifier and regressor)
- KD-Tree spatial index
- Ball Tree spatial index (in progress)
- Matrix and Vector math primitives

## Tech stack

- **Language:** Modern C++ (targeting high-performance idioms)
- **Build system:** CMake, with `FetchContent` for external dependencies
- **Test framework:** GoogleTest (with GoogleMock)
- **Math oracle (test-only):** Eigen, used for differential testing — not integrated into the library itself
- **Reference baseline:** scikit-learn, for informal correctness comparisons

The library itself is built as a `STATIC` library with a `jmll::jmll` alias target.

## Hard rules — read carefully

These rules are non-negotiable. They exist because the entire point of this project is for me to learn C++ by writing every line of the core library myself. AI assistance is welcome for everything *around* the code, but not for the code itself.

### Read-only directories

The following directories are **read-only**. You may read any file in them for context, but you must **never** create, modify, delete, move, or rename any file within them:

- `/include/`
- `/src/`
- `/tests/`

This includes — but is not limited to — `.cpp`, `.hpp`, `.h`, `.cc`, `.tpp`, `CMakeLists.txt` files inside these directories, and any other file at any depth beneath them. If a task would require touching a file in any of these directories, **stop and tell me**. Do not suggest workarounds that involve writing to these paths. Do not offer to "just sketch" a C++ file for me to copy-paste — I want to write it myself.

If I explicitly override this rule for a specific task in a specific message, that override applies **only** to that task and only to the specific files I name. The default state always reverts to read-only after the task is done.

### What you *can* do

You are encouraged to help with everything outside those three directories, including:

- GitHub Actions workflows (`.github/workflows/`)
- Issue and PR templates (`.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`)
- Repo configuration files (`.clang-format`, `.clang-tidy`, `.gitignore`, `.editorconfig`, `CMakePresets.json` at the repo root)
- Top-level `CMakeLists.txt` *only if the change does not affect how files in `/include`, `/src`, or `/tests` are compiled*
- Documentation (`README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/`)
- Doxygen configuration
- Build, lint, format, and benchmark scripts
- Reading any file in the repo (including the read-only directories) to understand context
- Suggesting changes to C++ code in chat as **discussion only** — never as file edits

## Working style

- Default to high-level explanations before diving into implementation detail. I learn best when I understand the *why* before the *how*.
- Be opinionated. If I propose something and there's a better approach, say so and explain the tradeoff.
- For non-trivial infrastructure changes, walk me through the design before writing the file.
- Conventional Commits is the commit style for this repo (`feat:`, `fix:`, `chore:`, `ci:`, `docs:`, `refactor:`, `perf:`, `test:`).
- Branch naming: `<type>/<issue-number>-<short-description>`, e.g. `chore/12-issue-templates`.

## Context about me

I'm a third-year Computer Science / Mathematics student at the University of Queensland, focused on high-performance computing and machine learning. I'm building this project to be portfolio-grade for quant developer roles, so decisions should be made with that audience in mind — clean architecture, rigorous testing, modern C++ idioms, and the ability to talk confidently about every design choice in an interview.
