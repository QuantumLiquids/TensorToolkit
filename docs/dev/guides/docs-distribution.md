# Documentation Distribution

This guide describes ways to distribute TensorToolkit documentation to
downstream projects.

## Option 1: Doxygen tag file (recommended)

```bash
cd docs
./build_docs.sh
```

This generates `docs/build/TensorToolkit.tag` (and HTML under
`docs/build/html/`). Downstream projects can reference this tag file to link
into TensorToolkit API pages.

Example Doxygen config (assuming the TensorToolkit checkout lives at
`../TensorToolkit`):

```
TAGFILES = ../TensorToolkit/docs/build/TensorToolkit.tag=../TensorToolkit/docs/build/html
```

## Option 2: HTML bundle

```bash
cd docs/build
tar -czf TensorToolkit-docs.tar.gz html/
```

Use this for offline distribution or attaching to releases.

## Option 3: API summary snapshot

There is no built-in tool for this yet. If you need a snapshot, generate the
HTML with Doxygen and export a minimal index from `docs/build/html/` using your
preferred tooling.

## Notes

- Keep documentation bundles versioned alongside releases.
- Ensure relative links remain stable for downstream projects.
