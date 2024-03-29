import matplotlib
import polars as pl
import pandas as pd
import sys
import os
import scanpy as sc
import anndata as ad

sc.settings.verbosity = 2  # show logging output
sc.settings.autosave = True  # save figures, do not show them
sc.settings.set_figure_params(dpi=300)  # set sufficiently high resolution for saving
PROJECT_DIR = "/home/jovyan/share/data/analyses/benjamin/Single_cell_project_rapids"


def basic_analysis(proj_dir):
    grit_filt_df = pl.read_parquet(os.path.join(proj_dir, "Beactica/Results/grit/sc_grit_full_FILTERED.parquet"))
    print("Data imported")
    grit_filter_df_sampled_pd = grit_filt_df.to_pandas()
    features_fixed = [f for f in grit_filt_df.columns if "Feature" in f]
    meta_features = [col for col in grit_filter_df_sampled_pd.columns if col not in features_fixed]
    adata = ad.AnnData(X = grit_filter_df_sampled_pd[features_fixed], obs = grit_filter_df_sampled_pd[meta_features])
    print("Starting scanpy!")
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.paga(adata, groups = "compound_id")
    sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    sc.tl.umap(adata, init_pos='paga')
    print("Embedding complete. Starting clustering")
    #sc.tl.leiden(adata, key_added='clusters', resolution=0.2)
    #sc.tl.louvain(adata)

    adata.write('sc_embedding_scanpy_Beactica_fixed.h5ad')



if __name__ == "__main__":
    basic_analysis(PROJECT_DIR)
