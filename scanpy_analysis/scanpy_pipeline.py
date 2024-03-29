import matplotlib
import polars as pl
import pandas as pd
import sys
import os
import scanpy as sc
import anndata as ad
import click  # Import the click library
import re

sc.settings.verbosity = 2  # show logging output
sc.settings.autosave = True  # save figures, do not show them
sc.settings.set_figure_params(dpi=300)  # set sufficiently high resolution for saving



def is_meta_column(c):
    for ex in '''
        Metadata
        ^Count
        ImageNumber
        Object
        Parent
        Children
        Plate
        Well
        location
        Location
        _[XYZ]_
        _[XYZ]$
        Phase
        Scale
        Scaling
        Width
        Height
        Group
        FileName
        PathName
        BoundingBox
        URL
        Execution
        ModuleError
        LargeBrightArtefact
    '''.split():
        if re.search(ex, c):
            return True
    return False

def extract_features_deepprofiler(df):
    # Function to extract features and metadata for DeepProfiler
    features_fixed = [f for f in df.columns if "Feature" in f]
    meta_features = [col for col in df.columns if col not in features_fixed]
    return features_fixed, meta_features

def extract_features_cellprofiler(df):
    # Function to extract features and metadata for CellProfiler
    # Replace this with the actual implementation needed for CellProfiler
    meta_features = [col for col in df.columns if is_meta_column(col)]
    features_fixed = [feat for feat in df.columns if feat not in meta_features]
    return features_fixed, meta_features


@click.command()
@click.option('--input', '-i', 'input_dir', required=True, help='Input directory path merged with project root.')
@click.option('--output', '-o', 'output_file', required=True, help='Output file name (or directory).')
@click.option('--compound_col', '-c', 'compound_col', required=True, help='Name of the compound column for calculations.')
@click.option('--extraction-method', '-e', 'extraction_method', required=True, type=click.Choice(['deepprofiler', 'cellprofiler'], case_sensitive=False), help='Feature extraction method.')
def basic_analysis(input_dir, output_file, extraction_method, compound_col):
    # Merge the input directory with the project root
    full_input_path = input_dir
    
    grit_filt_df = pl.read_parquet(full_input_path)
    print("Data imported")
    grit_filter_df_sampled_pd = grit_filt_df.to_pandas()
    if extraction_method.lower() == 'deepprofiler':
        features_fixed, meta_features = extract_features_deepprofiler(grit_filter_df_sampled_pd)
    elif extraction_method.lower() == 'cellprofiler':
        features_fixed, meta_features = extract_features_cellprofiler(grit_filter_df_sampled_pd)
    else:
        click.echo("Feature extractor not valid. Choose 'deepprofiler' or 'cellprofiler'")
        sys.exit(1)
    adata = ad.AnnData(X=grit_filter_df_sampled_pd[features_fixed], obs=grit_filter_df_sampled_pd[meta_features])
    print("Starting scanpy!")
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.paga(adata, groups=compound_col)
    sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    sc.tl.umap(adata, init_pos='paga')
    print("Embedding complete. Saving file!")
    # sc.tl.leiden(adata, key_added='clusters', resolution=0.2)
    # sc.tl.louvain(adata)
    for col in adata.obs.columns:
        if issubclass(adata.obs[col].dtype.type, pd.api.types.pandas_dtype('object').type):
            adata.obs[col] = adata.obs[col].apply(lambda x: str(x) if pd.notnull(x) else x)
    adata.write(output_file)

if __name__ == "__main__":
    basic_analysis()