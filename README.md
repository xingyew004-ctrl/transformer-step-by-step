# transformer-step-by-step-pytorch

Step-by-step PyTorch implementation of core Transformer blocks for learning and experimentation.

## Overview

This repository contains a minimal, educational implementation of several core Transformer building blocks in PyTorch.

Currently implemented modules:

- Positional Encoding
- Multi-Head Attention
- Position-wise Feed-Forward Network
- Encoder Layer
- Decoder Layer

This project is intended for learning, code reading, and small experiments.

## Project Status

**Runnable PoC / educational example**

This repository currently demonstrates core Transformer modules and a simple shape-check demo.
It is **not** a full training framework or production-ready implementation.

## What Is Included

- `PositionalEncoding`
- `MultiHeadAttention`
- `PositionWiseFeedForward`
- `EncoderLayer`
- `DecoderLayer`

## What Is Not Included Yet

- Full Transformer model wrapper
- Embedding layers
- Output projection / generator head
- Training loop
- Inference pipeline
- Dataset processing
- Automated tests beyond shape smoke tests
- Benchmark results

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- .gitignore
|-- requirements.txt
|-- src/
|   `-- transformer_step_by_step/
|       |-- __init__.py
|       `-- modules.py
|-- examples/
|   `-- basic_demo.py
|-- tests/
|   `-- test_shapes.py
`-- .github/
    |-- ISSUE_TEMPLATE/
    |   |-- bug_report.md
    |   `-- feature_request.md
    `-- PULL_REQUEST_TEMPLATE.md
```

## Requirements

- Python: `[TO_BE_FILLED]`
- PyTorch: `[TO_BE_FILLED]`

At minimum, this project requires:

```bash
pip install torch
```

## Installation

```bash
git clone https://github.com/<your-username>/transformer-step-by-step-pytorch.git
cd transformer-step-by-step-pytorch
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the demo script:

```bash
python examples/basic_demo.py
```

## Example Output

```text
PositionalEncoding output: torch.Size([2, 4, 8])
MultiHeadAttention output: torch.Size([2, 4, 8])
FFN output: torch.Size([2, 4, 8])
EncoderLayer output: torch.Size([2, 4, 8])
DecoderLayer output: torch.Size([2, 4, 8])
```

## Known Limitations

- The repository currently contains module-level implementations only.
- There is no end-to-end Transformer class yet.
- No training or evaluation pipeline is included.
- No correctness check against reference implementations is included.

## Roadmap

- [x] Clean up duplicate imports and code style
- [x] Add `requirements.txt`
- [x] Add shape smoke tests
- [x] Move demo code into `examples/`
- [ ] Add a full Transformer wrapper module
- [ ] Add training and inference examples
- [ ] Add bilingual documentation (optional)

## Contributing

Issues and pull requests are welcome.
Please open an issue first for major changes.

## License

MIT License.
