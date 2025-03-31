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
Usage
===
We have uploaded two stGNN output files ("mu_gene_expression.csv" and "disp_gene_expression.csv") generated from Mouse Brain Anterior datasets analysis, which can be directly utilized by running the stgnn.py script. The corresponding spatial transcriptomics (ST) dataset and single-cell RNA-seq dataset for Mouse Brain Anterior are available for download at: https://zenodo.org/record/6925603#.YuM5WXZBwuU.
