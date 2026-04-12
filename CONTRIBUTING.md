# Contributing Guide

Thank you for your interest in contributing to OceanOSSE! This document explains how to set up your development environment, run tests, and submit changes.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming environment for contributors of all backgrounds and experience levels.

---

## Getting Started

### Prerequisites

This project uses [Pixi](https://pixi.sh) to manage dependencies and development tasks. Pixi handles all language runtimes and packages, so you do not need to install anything else manually.

Install Pixi by following the [official installation instructions](https://pixi.sh/latest/#installation):

```bash
# Linux / macOS
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```

### Cloning the Repository

```bash
git clone https://github.com/AMOCcommunity/OceanOSSE.git
cd OceanOSSE
```

### Setting Up the Environment

Install all project dependencies into an isolated Pixi environment:

```bash
pixi install
```

This reads `pixi.toml` and resolves all dependencies automatically. You do not need to activate any environment manually — Pixi handles this for each command.

---

## Development Workflow

All common development tasks are defined as Pixi tasks in `pixi.toml`. Run any task with:

```bash
pixi run <task-name>
```

Key tasks include:

| Task | Command | Description |
|------|---------|-------------|
| Unit Testing | `pixi run tests` | Run OceanOSSE unit tests using pytest |
| Ruff Linting | `pixi run lint` | Run code linting using Ruff |
| Ruff Linting | `pixi run fix` | Run code linting and auto-fix issues using Ruff |
| Ruff Formatting | `pixi run format` | Run code formatting using Ruff |
| Preview Docs | `pixi run preview-docs` | Run local preview of OceanOSSE Documentation |

To see all available tasks:

```bash
pixi task list
```

---

## Running Tests

Run the full unit test suite with:

```bash
pixi run tests
```

Please ensure all tests pass before opening a pull request. If you are adding new functionality, include corresponding tests.

---

## Code Style

This project follows standard code style conventions. If linting or formatting tasks are defined, run them before committing:

```bash
pixi run lint
pixi run format
```

---

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch from `main`:

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes**, following the code style guidelines above.

3. **Add or update tests** to cover your changes.

4. **Run the test suite** and confirm everything passes:

   ```bash
   pixi run tests
   ```

5. **Commit your changes** with a clear, descriptive commit message:

   ```bash
   git commit -m "Add support for XYZ"
   ```

6. **Push** your branch and open a pull request against `main`.

7. Fill in the pull request template, describing what changed and why.

A maintainer will review your pull request. Please be patient and responsive to any feedback.

---

## Reporting Issues

If you encounter a bug or have a feature request, please [open an issue](https://github.com/AMOCcommunity/OceanOSSE/issues) and include:

- A clear description of the problem or request
- Steps to reproduce the issue (for bugs)
- Your operating system and the output of `pixi info`
- Any relevant error messages or stack traces
