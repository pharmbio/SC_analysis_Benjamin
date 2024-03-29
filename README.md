# Single-cell morphological analysis - Benjamin Frey

This repository includes the main scripts for my master thesis and on-going project about single-cell morphological data.

Result dataframes and plots are not included in this repository!

### Supervised analysis

Mechanism of action classification for two SPECS5K subsets. Main training script:

```
mlp_classification.py 
```

Remaining scripts contain the evaluation of training results and statistical calculations. 

### Unsupervised analysis

Analysis scripts for unsupervised Beactica and SPECS5K analysis. Additionally includes normalization scripts, grit calculations, cell viewer as well as standard UMAP analysis.

### Scanpy analysis

Specific unsupervised analysis involving scanpy and cellxgene framework. Specifically, usage of PAGA embeddings and leiden/louvain clusterings. 
E distance and testing also included here.

### Single-cell viewer

Code base for single-cell bokeh application developped with Dan. Use
```
generate_single_cell_bokeh.py 
```

for  viewer .html with custom data sets.
