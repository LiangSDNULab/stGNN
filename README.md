# stGNN: spatially informed cell-type deconvolution based on deep graph learning and statistical model

Overview
===
![Image text](https://github.com/LiangSDNULab/stGNN/blob/main/stGNN.png)


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
