# stGNN: spatially informed cell-type deconvolution based on deep graph learning and statistical model

Overview
===
![Image text](https://github.com/LiangSDNULab/stGNN/blob/main/stGNN.png)
stGNN consists of three main modules: an auto-encoder module, an GCN-based encoder module,  and a predictor. First, the auto-encoder takes as inputs the preprocessed gene expressions to learn non-spatial representations, capturing the fundamental gene expression patterns. Next, the GCN-based encoder is to learn spatial representations by taking full advantages of gene expressions and spatial coordinate information. To learn more informative representations, an attention mechanism is introduced to adaptively integrate the non-spatial and spatial representations in a layer-wise way. The last layer of the GCN-based encoder is designed as a multiple classification layer (predictor) to predict cell type proportions. To align distribution of ST data and scRNA-seq data and facilitate model training, a negative log-likelihood loss function is introduced, where the mean and dispersion parameter of ST data is estimated by the scvi-tools package from scRNA-seq reference data.  

Requirements
===
```
Python >= 3.8
torch = 1.11.0,
scanpy = 1.10.3,
pandas = 2.2.3,
numpy = 1.26.4,
scipy = 1.10.0
scvi = 0.6.8
anndata = 0.10.8
```
