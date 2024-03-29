from IPython.display import HTML
from bokeh_scatter import bokeh_scatter, bokeh_scatter_with_selection
import pandas as pd
import numpy as np
from markupsafe import Markup
import viewer
import anndata as ad
import os
import click 

@click.command()
@click.option('--input_dataset','-i', type=click.Path(exists=True))
@click.option('--filter_col', '-f', type=str)
@click.option('--crop_size','-c', type=int, help='Cropping window used for extraction/view.')
@click.option('--output_dataset','-o', type=click.Path())
@click.option('--embedding_name','-e', type=str, default="X_umap", help='Embedding name from anndata.')

def generate_view(input_dataset, crop_size, output_dataset, filter_col, embedding_name):
    if "BF" in input_dataset:
        hues = f'Metadata_cmpdName moa project'.split()
    else:
        hues = f'Metadata_cmpdName moa_broad project'.split()
    df = prepare_bokeh_sc(input_dataset, crop_size, embedding_type = embedding_name)
    print(df.dtypes)
    plot_html = bokeh_scatter_with_selection(
        df,
        x = 'UMAP 1',
        y = 'UMAP 2',
        hues = hues,
        title='Single cell UMAP',
        hover_columns='Metadata_Plate Metadata_cmpdName Metadata_Well clip'.split(),
        # marker='square', size=0.95,
        size=5,
        filter_column= filter_col
    )
    save_html(plot_html, output_dataset)

def save_html(html_content, file_name):
    with open(file_name, "w") as file:
        file.write(html_content)

def prepare_bokeh_sc(path, box_size, embedding_type = "umap"):
    df = ad.read_h5ad(path)

    embedding_df = pd.DataFrame()

    # Search for matching keys in .obsm
    for key in df.obsm.keys():
        if embedding_type.lower() in key.lower():
            # If a match is found, convert the embedding to a DataFrame
            embedding_array = df.obsm[key]
            cols = [f"{embedding_type}_{i+1}" for i in range(embedding_array.shape[1])]
            embedding_df = pd.DataFrame(embedding_array, columns=cols, index=df.obs.index)
            break  # Stop searching after the first match

    # If an embedding was found, concatenate it with .obs
    if not embedding_df.empty:
        result_df = pd.concat([df.obs, embedding_df], axis=1)
    else:
        # If no embedding was found, return the original .obs DataFrame
        print(f"No embedding found for '{embedding_type}'. Returning original .obs DataFrame.")
        result_df = df.obs

    result_df["site"] = result_df["Metadata_Site"].str[-1].astype(int)
    result_df["well"] = result_df["Metadata_Well"]
    if "barcode" not in result_df.columns:
        result_df["barcode"] = result_df["Metadata_Plate"]
    result_df["clip"] = result_df.apply(lambda row: viewer.ClipSquare(row['Nuclei_Location_Center_X'], 
                                                                  row['Nuclei_Location_Center_Y'], 
                                                                  box_size).to_str(), axis=1)
    result_df["UMAP 1"] = result_df[f"{embedding_type}_{1}"]
    result_df["UMAP 2"] = result_df[f"{embedding_type}_{2}"]
    result_df = result_df.reset_index(drop = True)
    if "name_0" in result_df.columns:
        del result_df['name_0']
    num_cols = ["Metadata_cmpdConc", "cells_per_well", "grit", "group"]

    for col in result_df.columns:
        # Ensure you reference result_df here, not df
        if col in num_cols:
            # Convert specified columns to numerical
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        elif pd.api.types.is_categorical_dtype(result_df[col]):
            # Convert categorical columns to string, except those specified for numerical conversion
            result_df[col] = result_df[col].astype(str)
    #result_df = result_df[:1000]
    return result_df



if __name__ == '__main__':
    generate_view()
