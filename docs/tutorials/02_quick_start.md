# Quick Start: Z2 Ising Tensor Network and TRG

Zero-to-TRG in a couple of minutes.

### Option A: Jam with an AI buddy

- Make sure TensorToolkit is installed.
- Open your favorite AI coding buddy (Cursor, Claude, Qwen, etc.).
- Paste this prompt and roll:

```text
Write a minimal Z2 Ising TRG demo using TensorToolkit:
- Create Z2 indices
- Build the Ising vertex tensor at inverse temperature β
- Perform one TRG coarse-graining step: contract → reshape → SVD → truncate → reassemble
- Print the incremental log Z (free-energy contribution)
- Keep it concise, compile-ready, and easy to tweak
```

- **Important**: have a lunch or grab a coffee while AI works. Then check the results.

### Option B: Skim the code, learn by doing

This is the quickest way I think to get a feel for the toolkit:
- Ready-made examples live in `examples/z2_ising_tensor.cpp` and `examples/z2_ising_trg.cpp`.
- Read the code first, run it, then tweak β, χ (bond cut), and the number of steps.
- Jump into the matching tutorial chapter when you hit a question.

Prefer details? The tutorials have your back (and your GPT’s). 
See the next chapters for indices, tensors, contractions, decompositions, and the full TRG walkthrough.
