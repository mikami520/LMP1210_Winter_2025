# Instruction for Assignment 3

## Install the required packages

```bash
conda create -n a3 python=3.10 -y
conda activate a3
pip install scikit-learn pandas numpy matplotlib seaborn umap-learn torch torchvision torchaudio
```

## Run the code

- `A3_code.py` is the main code for this assignment
- `autoencoder.py` is the file that contains the autoencoder model.

```bash
python A3_code.py \
--train # if need to train the model
--pretrain # path to the pretrained autoencoder model, provided by **autoencoder_best.pth**. If set to train, this will be ignored. 
```

## Expected output

- `Q2.pdf` (Answer to Q2)
- `Q4Pc_largePC.png`
- `Q4Pc_smallPC.png`
- `Q4PdBottom.png`
- `Q4PdTop.png`
- `Q4Pe.png`
- `Q5Pa.png`
- `Q5Pb.png`
- `Q5Pc.png`
