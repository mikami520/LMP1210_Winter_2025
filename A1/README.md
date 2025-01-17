# Instruction for Assignment 1

## Install the required packages

```bash
conda create -n a1 python=3.10 -y
conda activate a1
pip install scikit-learn pandas numpy matplotlib graphviz
```

For MacOS, you may need to install `graphviz` using `brew install graphviz`.

For Linux, you may need to install `graphviz` using `sudo apt-get install graphviz`.

## Run the code

```bash
python A1_code.py
```

## Expected output

- `knn_accuracy_cosine.png`
- `knn_accuracy_minkowski.png`
- `dt_min_sample_leaf_1.png`
- `dt_min_sample_leaf_2.png`
- `dt_min_sample_leaf_3.png`
