# Contributing to CR-PH-CBF

Thank you for your interest in contributing to CR-PH-CBF! This document provides guidelines for contributing to the project.

## 🎯 How to Contribute

We welcome contributions in the following areas:

- 🐛 **Bug fixes** - Report or fix bugs
- ✨ **New features** - Add new control methods, environments, or models
- 📚 **Documentation** - Improve docs, tutorials, or examples
- 🧪 **Tests** - Add or improve test coverage
- 🎨 **Code quality** - Refactoring, optimization, style improvements

## 📋 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/CR-PH-CBF.git
cd CR-PH-CBF
git remote add upstream https://github.com/ORIGINAL_OWNER/CR-PH-CBF.git
```

### 2. Set Up Development Environment

```bash
# Create conda environment
conda create -n crphcbf-dev python=3.8
conda activate crphcbf-dev

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 💻 Development Guidelines

### Code Style

We follow PEP 8 and use automatic formatters:

```bash
# Format code with black
black src/ tests/ scripts/

# Sort imports with isort
isort src/ tests/ scripts/

# Check style with flake8
flake8 src/ tests/ scripts/
```

### Documentation

- Add docstrings to all public functions and classes (Google style)
- Update README.md if adding new features
- Add examples to `docs/tutorials/` for significant features

Example docstring:

```python
def my_function(param1: int, param2: str) -> float:
    """
    Brief description of function.
    
    Longer description if needed, explaining what the function does,
    when to use it, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> result = my_function(5, "test")
        >>> print(result)
        15.0
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return float(param1 * len(param2))
```

### Testing

All new code should include tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Check code coverage
pytest --cov=src --cov-report=html
```

**Test guidelines:**
- Write unit tests for individual functions/classes
- Write integration tests for end-to-end workflows
- Aim for >80% code coverage
- Use fixtures for common setup
- Mock external dependencies

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(controllers): add robust MPC baseline

Implement Model Predictive Control baseline for comparison
with CBF methods. Includes tube-based robust MPC with
uncertainty sets.

Closes #42
```

## 🔄 Pull Request Process

### 1. Before Submitting

- ✅ All tests pass: `pytest tests/`
- ✅ Code is formatted: `black .` and `isort .`
- ✅ No linting errors: `flake8 src/ tests/`
- ✅ Documentation is updated
- ✅ CHANGELOG.md is updated (if applicable)

### 2. Submit Pull Request

1. Push your branch to your fork
2. Open a PR against the `main` branch
3. Fill out the PR template completely
4. Link any related issues

### 3. PR Review

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged!

### PR Checklist

```markdown
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Added tests for new functionality
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts
```

## 🐛 Reporting Bugs

Use GitHub Issues with the bug template:

**Include:**
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, package versions)
- Code snippet demonstrating the issue
- Error messages and stack traces

## 💡 Requesting Features

Use GitHub Issues with the feature template:

**Include:**
- Clear description of the feature
- Use case and motivation
- Proposed API or interface
- Alternatives considered
- Willing to contribute implementation?

## 📝 Adding Experiments

When adding new experiments:

1. Create config file in `configs/`
2. Add script in `scripts/`
3. Document in `docs/tutorials/`
4. Add expected results to `paper/data/`
5. Include visualization in `paper/figures/`

## 🧪 Adding New Methods

To add a new control method:

1. Implement in `src/crphcbf/controllers/`
2. Add tests in `tests/test_controllers.py`
3. Add to evaluation in `scripts/reproduce_paper_results.py`
4. Document in README and tutorials
5. Include comparison results

## 📚 Documentation Structure

```
docs/
├── installation.md          # Installation instructions
├── quickstart.md            # Quick start guide
├── api/                     # API reference (auto-generated)
├── tutorials/               # Step-by-step tutorials
│   ├── 01_basic_usage.md
│   ├── 02_custom_envs.md
│   └── 03_new_methods.md
└── faq.md                   # Frequently asked questions
```

## 🤝 Code Review Guidelines

For reviewers:

- ✅ Check that code follows style guidelines
- ✅ Verify tests cover new functionality
- ✅ Ensure documentation is clear and complete
- ✅ Run experiments to verify results
- ✅ Check for potential performance issues
- ✅ Verify backward compatibility

## 📧 Getting Help

- 💬 GitHub Discussions for questions
- 🐛 GitHub Issues for bugs
- 📧 Email maintainers for sensitive issues

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

Thank you to all our contributors! Your work makes this project better.

See [AUTHORS.md](AUTHORS.md) for a list of contributors.

---

**Questions?** Open a discussion on GitHub or email the maintainers.
