# Development Workflow

This guide outlines the development process and coding standards for the project.

## Branch Management

1. **Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Keep Branch Updated**
```bash
git fetch origin
git rebase origin/main
```

## Code Quality

Before committing changes, ensure code quality:

```bash
# Format code
black .

# Sort imports
isort .

# Run linting
flake8

# Run type checking
mypy .

# Run tests
pytest
```

## Commit Guidelines

1. **Write Clear Commit Messages**
```bash
# Format: <type>(<scope>): <description>
git commit -m "feat(agents): add new visualization capabilities"
git commit -m "fix(database): resolve connection timeout issue"
```

Common types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `style`: Formatting changes
- `chore`: Maintenance tasks

2. **Keep Commits Focused**
- One logical change per commit
- Break large changes into smaller commits
- Include tests with feature changes

## Pull Request Process

1. **Create Pull Request**
- Push your branch
- Create PR through GitHub
- Fill out PR template

2. **PR Requirements**
- All tests passing
- Code quality checks passing
- Documentation updated
- PR description complete

3. **Review Process**
- Address review comments
- Request re-review after changes
- Squash commits if needed

## Continuous Integration

All PRs must pass:
- Unit tests
- Integration tests
- Code quality checks
- Type checking
- Security scans

See [Testing Guide](./testing.md) for more details. 