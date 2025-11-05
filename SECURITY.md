# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of MorphML seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by:

1. **Email**: Send details to the project maintainers (create an issue with "SECURITY" in the title if no email is available)
2. **Private disclosure**: Use GitHub's private vulnerability reporting feature if available

### What to Include

When reporting a vulnerability, please include:

- **Description**: A clear description of the vulnerability
- **Impact**: What could an attacker do with this vulnerability?
- **Reproduction**: Step-by-step instructions to reproduce the issue
- **Environment**: Affected versions, Python version, OS, etc.
- **Proof of Concept**: Sample code or exploit demonstrating the vulnerability (if applicable)
- **Suggested Fix**: If you have ideas for how to fix it (optional)

### What to Expect

After you submit a vulnerability report:

1. **Acknowledgment**: We'll acknowledge receipt within 48 hours
2. **Assessment**: We'll assess the vulnerability and determine its severity
3. **Fix Development**: We'll work on a fix if the vulnerability is confirmed
4. **Disclosure Timeline**: We'll coordinate disclosure timing with you
5. **Credit**: We'll credit you in the security advisory (unless you prefer to remain anonymous)

### Disclosure Policy

- **Security fixes** will be released as soon as possible
- **Security advisories** will be published on GitHub
- **Coordinated disclosure**: We prefer a 90-day coordinated disclosure timeline

## Security Best Practices

When using MorphML:

### Input Validation

- Always validate and sanitize user inputs
- Be cautious when loading models or data from untrusted sources
- Use appropriate data validation for configuration files

### Dependency Security

- Keep dependencies up to date
- Regularly run security audits: `poetry audit`
- Monitor security advisories for dependencies

### Model Security

- Verify the integrity of pre-trained models before loading
- Be cautious with pickle files from untrusted sources
- Use secure storage for sensitive model data

### API Security

- Use authentication and authorization when exposing APIs
- Implement rate limiting to prevent abuse
- Validate all API inputs

### Environment Security

- Use virtual environments for isolation
- Don't hardcode secrets or API keys in code
- Use environment variables or secure vaults for sensitive data
- Don't commit `.env` files or credentials to version control

## Known Security Considerations

### Pickle Usage

MorphML may use Python's `pickle` module for model serialization. Be aware that:

- **Never load pickle files from untrusted sources**
- Pickle can execute arbitrary code during deserialization
- Consider using safer alternatives like `safetensors` for production

### Dynamic Code Execution

Some features may involve dynamic code execution:

- Be cautious with user-provided code or expressions
- Implement proper sandboxing for untrusted code
- Validate and sanitize inputs thoroughly

## Security Updates

Security updates will be:

- Released as patch versions (e.g., 0.1.1)
- Documented in the CHANGELOG
- Announced through GitHub security advisories
- Tagged with `security` label in releases

## Scope

This security policy applies to:

- The core MorphML library
- Official examples and documentation
- Build and deployment scripts

It does not cover:

- Third-party dependencies (report to respective maintainers)
- User-created models or applications using MorphML
- Deployment infrastructure (user's responsibility)

## Questions

If you have questions about this security policy or MorphML's security:

- Open a public issue for general security questions (not vulnerabilities)
- Contact maintainers for specific security concerns

## Recognition

We appreciate security researchers who help keep MorphML safe:

- Responsible disclosure will be acknowledged in security advisories
- We'll credit researchers who report valid vulnerabilities (unless anonymous)
- Significant contributions may be recognized in project documentation

Thank you for helping keep MorphML and its users safe!
