# MorphML v1.0.0 - PyPI Publishing Checklist

## ✅ Pre-Publication Checklist

### 1. Code Quality (COMPLETED ✅)
- [x] All tests pass
- [x] Black formatting applied
- [x] Ruff linting clean
- [x] Mypy type checking passes
- [x] No pytest collection errors

### 2. Package Metadata (COMPLETED ✅)
- [x] Version set to 1.0.0
- [x] Authors and maintainers updated
- [x] License specified (MIT)
- [x] README.md complete
- [x] CHANGELOG.md updated
- [x] Keywords and classifiers set
- [x] Repository and documentation URLs set

### 3. Build Verification (COMPLETED ✅)
- [x] Package builds successfully: `poetry build`
- [x] Distribution files created in `dist/`

## 🚀 Publishing Steps

### Step 1: Get PyPI API Token

1. Go to https://pypi.org/account/register/ (if you don't have an account)
2. Verify your email
3. Go to https://pypi.org/manage/account/
4. Scroll to "API tokens" section
5. Click "Add API token"
6. Name: "MorphML Publishing"
7. Scope: "Entire account" (or create project-specific after first upload)
8. Copy the token (starts with `pypi-`)

### Step 2: Configure Poetry with Token

```bash
poetry config pypi-token.pypi pypi-YOUR_TOKEN_HERE
```

**IMPORTANT:** Replace `pypi-YOUR_TOKEN_HERE` with your actual token

### Step 3: Run the Publish Script

```bash
./publish.sh
```

This script will:
1. Clean previous builds
2. Build the package
3. Test the package in a temporary environment
4. Publish to PyPI
5. Provide next steps

### Alternative: Manual Publishing

If you prefer manual control:

```bash
# Clean
rm -rf dist/ build/ *.egg-info

# Build
poetry build

# Publish
poetry publish
```

## 📦 Post-Publication Steps

### 1. Verify Installation (Wait 2-3 minutes)

```bash
# In a new environment
pip install morphml

# Test import
python -c "import morphml; print(morphml.__version__)"
```

### 2. Create GitHub Release

```bash
git tag -a v1.0.0 -m "Release v1.0.0 - First Stable Release"
git push origin v1.0.0
```

Then go to GitHub and create a release from the tag with the CHANGELOG content.

### 3. Update Documentation

- Update ReadTheDocs (if configured)
- Update README badges
- Announce on social media/forums

## 🔒 Security Notes

- **NEVER** commit your PyPI token to git
- Store token securely (password manager)
- Use project-scoped tokens when possible
- Rotate tokens periodically

## 📊 Package Information

- **Package Name:** morphml
- **Version:** 1.0.0
- **PyPI URL:** https://pypi.org/project/morphml/
- **Install Command:** `pip install morphml`

## 🎯 Success Criteria

✅ Package appears on PyPI
✅ Can install with `pip install morphml`
✅ Import works: `import morphml`
✅ Version correct: `morphml.__version__ == "1.0.0"`

## 🆘 Troubleshooting

### "Package already exists"
- Version 1.0.0 is already published
- Bump version: `poetry version patch` (→ 1.0.1)
- Update CHANGELOG.md
- Rebuild and republish

### "Invalid token"
- Check token is correct
- Ensure no extra spaces
- Token must start with `pypi-`
- Reconfigure: `poetry config pypi-token.pypi pypi-NEW_TOKEN`

### "Package name taken"
- morphml should be available
- If not, choose alternative name in pyproject.toml

## 📞 Support

- GitHub Issues: https://github.com/TIVerse/MorphML/issues
- Email: vedanth@vedanthq.com, eshanized@proton.me
