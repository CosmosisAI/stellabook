---
paths:
  - "**/*.py"
---

# Python Style

Prioritize idiomatic Python. Code should emphasize correctness, maintainability, and testability.

## Types and Data Modeling

- Favor strong types: proper classes, enums, and typed datastructures over raw strings or dicts
- Use enums for fixed sets of values (configuration constants, mode selectors, etc.)
- Use frozen dataclasses or Pydantic models for data — prefer immutable interfaces
- Maintain clear module-level separation boundaries
- Emphasize modular design patterns and composable interfaces

## Control Flow and Structure

- Favor `match` statements over `if`/`elif` chains when dispatching on values
- Write code as data processing pipelines: pass inputs through composed functions rather than mutating shared state
- Aim for a declarative/functional style where practical
- Keep code flat — avoid deep nesting. Early returns over nested conditionals
- Keep functions short and focused on a single responsibility
- Apply DRY — extract shared logic rather than duplicating it
- Favor stdlib modules for common patterns (`pathlib` for file I/O, `functools` for higher-order functions, `itertools` for iteration, `math` for math, etc.) rather than re-inventing logic

## Naming

- Use descriptive names for functions and variables — clarity over brevity
- `snake_case` for functions and variables
- `PascalCase` for classes
- `ALL_CAPS` for module-level constants
