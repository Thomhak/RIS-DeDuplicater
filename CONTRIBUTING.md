# Contributing to RIS Deduplicator

Thank you for your interest in contributing to RIS Deduplicator! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Check existing issues** first to avoid duplicates
2. **Create a new issue** with a clear title and description
3. **Include example files** (if possible) that demonstrate the problem
4. **Specify your Python version** and operating system

### Submitting Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** with clear, descriptive commit messages
3. **Test your changes** with various RIS files
4. **Update documentation** if you're adding new features
5. **Submit a pull request** with a clear description of your changes

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ris-deduplicator.git
cd ris-deduplicator

# Create a new branch for your feature
git checkout -b feature-name

# Make your changes and test them
python3 ris_deduplicator.py test_files/*.ris

# Commit your changes
git add .
git commit -m "Add feature: descriptive message"

# Push to your fork and submit a pull request
git push origin feature-name
```

## Code Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use descriptive variable and function names
- Add type hints where appropriate
- Include docstrings for functions and classes

### Testing

- Test with various RIS file formats and sources
- Test edge cases (empty files, malformed records, encoding issues)
- Ensure backward compatibility with existing functionality

### Documentation

- Update README.md for new features
- Include clear examples in docstrings
- Update help text for new command-line options

## Areas for Contribution

### High Priority
- **Additional duplicate detection methods** (DOI-based, fuzzy title matching)
- **Performance optimizations** for very large files
- **Better error handling** for malformed RIS files
- **Unit tests** for core functionality

### Medium Priority
- **GUI interface** for non-technical users
- **Configuration file support** for default settings
- **Export statistics** to CSV or JSON
- **Integration with reference managers**

### Low Priority
- **Docker container** for easy deployment
- **Web interface** for online processing
- **Batch processing scripts** for multiple directories
- **Plugin system** for custom duplicate detection algorithms

## Submitting Changes

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Changes have been tested with various RIS files
- [ ] Documentation has been updated
- [ ] Commit messages are clear and descriptive
- [ ] No breaking changes to existing functionality
- [ ] New features include appropriate error handling

### Review Process

1. **Automated checks** will run on your pull request
2. **Maintainer review** will check code quality and functionality
3. **Testing** with various RIS file formats
4. **Merge** after approval and successful tests

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions about usage or development
- **Email**: For security issues or private questions

## Code of Conduct

This project follows a simple code of conduct:

- Be respectful and constructive
- Focus on what's best for the project and community
- Show empathy towards other contributors
- Accept constructive criticism gracefully

Thank you for contributing to RIS Deduplicator!
