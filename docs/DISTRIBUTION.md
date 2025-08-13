# Distributing TensorToolkit Documentation (In Vibe Code)

This guide explains how to provide TensorToolkit documentation to projects that depend on it. 

## Option 1: Tag Files (Recommended for API Documentation)

### Generate Tag File
```bash
cd docs
./build_docs.sh
# This creates TensorToolkit.tag in the build directory
```

### Use in Vibe Code
1. **Copy the tag file** to your Vibe Code project
2. **Configure Doxygen** to reference it:

```cmake
# In Vibe Code's CMakeLists.txt
set(TAGFILES "${TENSORTOOLKIT_DOCS_DIR}/TensorToolkit.tag=../TensorToolkit")
```

```doxygen
# In Vibe Code's Doxyfile
TAGFILES = TensorToolkit.tag=../TensorToolkit
```

**Result**: Vibe Code documentation will have clickable links to TensorToolkit API functions, classes, and types!

## Option 2: HTML Documentation Bundle

### Create Bundle
```bash
cd docs/build
tar -czf TensorToolkit-docs-${VERSION}.tar.gz html/
```

### Distribute
- Include in TensorToolkit releases
- Host on project website
- Bundle with Vibe Code for offline access

## Option 3: API Summary Files

### Generate Markdown Summary
```bash
# Extract key API information
cd docs
doxygen -x Doxyfile | grep -E "(CLASS|FUNCTION)" > api_summary.md
```

### Include in Vibe Code
- Copy `api_summary.md` to Vibe Code docs
- Reference in project documentation
- Use for quick API lookup

## Best Practices

1. **Version Tagging**: Always include version information in documentation bundles
2. **Cross-References**: Use tag files for proper API linking
3. **Offline Access**: Provide self-contained documentation when possible
4. **Regular Updates**: Update documentation with each TensorToolkit release

## Vibe Code Integration (Step-by-Step)

For Vibe Code developers:

```bash
# 1. Build TensorToolkit docs
cd TensorToolkit/docs
./build_docs.sh

# 2. Copy tag file to Vibe Code
cp build/TensorToolkit.tag /path/to/VibeCode/docs/

# 3. Configure Vibe Code Doxyfile
echo "TAGFILES = TensorToolkit.tag=../TensorToolkit" >> VibeCode/docs/Doxyfile

# 4. Build Vibe Code docs with TensorToolkit references
cd /path/to/VibeCode/docs
doxygen Doxyfile
```

**What you get**: When viewing Vibe Code documentation, clicking on any TensorToolkit function/class will take you directly to the detailed TensorToolkit API documentation!

## Alternative: Direct HTML Integration

If you prefer to embed TensorToolkit docs directly in Vibe Code:

```bash
# Copy the entire HTML documentation
cp -r TensorToolkit/docs/build/html VibeCode/docs/tensortoolkit-api

# Reference in Vibe Code docs
echo "See [TensorToolkit API Documentation](tensortoolkit-api/index.html) for detailed function references."
```

This provides seamless API documentation integration while maintaining project independence. **Your developers will thank you for this!**
