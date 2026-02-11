# Contributing to Kidney Tumor Classification

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/kidney-tumor-classification.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit: `git commit -m "Description of changes"`
7. Push: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/
```

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Add docstrings to all functions and classes
- Write descriptive commit messages

## Testing

- Add tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage

## Pull Request Process

1. Update README.md with details of changes if applicable
2. Update the documentation with any new functionality
3. The PR will be merged once you have the sign-off of the maintainers

## Reporting Bugs

Use GitHub Issues to report bugs. Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

## Feature Requests

We welcome feature requests! Please:
- Check if the feature has already been requested
- Provide clear use case and benefits
- Be open to discussion

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards other community members

## Questions?

Feel free to open an issue or contact the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
