Below is example documentation you can place in this repository’s main **README.md** to guide users through both **training** a model and **solving** argumentation problems using AFGCN. Adjust or expand any sections as needed.

---

# AFGCN

AFGCN is an approximate abstract argumentation solver based on a Graph Convolutional Network (GCN) architecture. The repository provides two major workflows:

1. **Training**: Generating or refining the GCN models on a dataset of argumentation frameworks.
2. **Solving**: Using the trained models (and a baseline grounded solver) to answer credulous or skeptical decision queries for various abstract argumentation semantics.

---

## Prerequisites

- **Python 3.6+** 
- **Pytorch**
- **DGL** and **DGLGO** (for graph operations)
- **scikit-learn** (for data utilities)

Install core dependencies (adjust the Python package links or versions as appropriate):

```bash
pip install torch
pip install scikit-learn
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

---

## 1. Training

### 1.1 Dataset

AFGCN expects a dataset of argumentation frameworks in a suitable textual form (each file typically specifying the number of arguments and the directed attacks). You can download or clone the dataset from [lmlearning/AFGraphLib](https://github.com/lmlearning/AFGraphLib) (or your chosen path/dataset). Place or link these dataset files somewhere locally so they can be accessed by the training script.

### 1.2 Usage

Inside the **`Training`** directory is `train.py`. This script shows how to train the GCN model on the dataset. You may edit or extend it to match your data’s location, batch sizes, or training hyperparameters. Example usage:

```bash
cd Training
python train.py \
    --data_dir /path/to/AFGraphLib \
    --epochs 50 \
    --learning_rate 0.001 \
    --out_dir ./checkpoints
```

*Arguments above are placeholders—update to match how `train.py` processes command-line options.*

**Key points**:

- **`--data_dir`**: Directory containing your argumentation framework data (the AFGraphLib, for example).
- **`--epochs`**: Number of training epochs.
- **`--learning_rate`**: Learning rate for the optimizer.
- **`--out_dir`**: Where to store the trained `.pth` model checkpoints.

After training, you should have new `.pth` files representing the trained models for different semantics.

---

## 2. Solving

The **`Solver/`** folder contains:

- Multiple `.pth` files (e.g., `DC-CO.pth`, `DS-STG.pth`, etc.)—trained model checkpoints that correspond to different semantics (Complete, Preferred, Stable, etc.).
- A reference implementation (`solver.py` and `solver.sh`) that conforms to the ICCMA approximate track interface.
- A JSON file (`thresholds.json`) that provides probability thresholds used by each semantic.

### 2.1 Dependencies

Ensure the same core dependencies (PyTorch, DGL, etc.) are installed. See [Prerequisites](#prerequisites) above.

### 2.2 Usage Overview

You can run the solver in an ICCMA-like manner using `solver.sh`:

```bash
cd Solver
./solver.sh -p <problem> -f <file> -a <argument>
```

where:

- **`<problem>`**: The argumentation problem type, e.g., `DC-CO`, `DS-PR`, `DC-ST`, `DS-STG`, etc.
- **`<file>`**: Path to the input file (in `.tgf`, `.apx`, or similarly specified format).
- **`<argument>`**: The query argument (for “decide credulously” or “decide skeptically” tasks).

For example:

```bash
./solver.sh -p DS-ST -f myAF.tgf -a 2
```

The solver will output **YES** or **NO** depending on the acceptance status of argument 2 under the DS-ST (decide skeptically under the Stable semantics) approximation.

### 2.3 Detailed Steps

1. **Add Execute Permission** (if needed):
   ```bash
   chmod +x solver.sh
   ```
2. **Run** the solver as shown above. The script internally calls `solver.py`, which:
   - Reads the AF from `<file>`.
   - If the grounded solver’s cascade labels the queried argument as “in,” it prints **YES** immediately.
   - Otherwise, it loads the relevant `.pth` model (based on `<problem>`), calculates graph-based features, then applies a GCN to decide acceptance or not, printing **YES** or **NO**.
3. **Threshold Tuning**: If you want to tweak probability thresholds for acceptance, edit `thresholds.json` accordingly. Each problem key in the JSON maps to the acceptance cutoff (e.g., `0.9` means an argument with predicted probability above 0.9 is considered “accepted”).

---

## Additional Notes

- **Paper**: A short LaTeX paper discussing AFGCN is in `Solver/paper/`.
- **Testing**: A few example argumentation files are in `Solver/testaf*.txt` and `Solver/test_set.txt` for demonstration.
- **Performance**: The solver is designed for *approximate* solutions and may differ from exact results on certain complex frameworks. It is optimized for quick approximate answers in competition or large-scale usage scenarios.
- **Extensibility**: You can add or modify semantics by introducing new `.pth` models, referencing them in `thresholds.json`, and calling them with new problem keys in `solver.sh`.

---

**License & Contributions**:

Contributions, bug reports, or feature requests are welcome. Please see `LICENSE` for details, and open issues or pull requests for improvements.

Enjoy experimenting with AFGCN—an approximate solver that leverages the power of GCNs in abstract argumentation!
