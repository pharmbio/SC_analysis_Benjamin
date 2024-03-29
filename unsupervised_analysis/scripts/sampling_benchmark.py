import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn.metrics

import scipy.linalg
import scipy.spatial.distance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('ytick', labelsize=7)
import seaborn as sns
import os
import psutil
import polars as pl
import cuml
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler
import numpy as np 
from collections import OrderedDict
import math
import umap
import time
import torch



def main():
    print("Importing data")
    mad_norm_df = pl.read_parquet('sc_profiles_normalized_SPECS3K.parquet')
    mad_norm_df = mad_norm_df.filter(pl.col('Metadata_Plate') != "P101384")
    features_fixed = [f for f in mad_norm_df.columns if "Feature" in f]
    #grit_filt_df = load_grit_data("grit_parquet_specs3k", plates)
    grit_filt_df = pl.read_parquet("grit_parquet_specs3k/sc_grit_FULL.parquet")
    grit_filt_df = grit_filt_df.filter((pl.col("grit") > 1.5) | (pl.col("Metadata_cmpdName") == "[DMSO]"))
    dat = sample_compounds(mad_norm_df, grit_filt_df, sampling_rate = 1, mode = "normal")
    
    compounds = list(mad_norm_df["Metadata_cmpdName"].unique())
    n_neighb = [10, 50, 100, 150, None]
    min_dist = [0.01, 0.05, 0.1, 0.3, 0.5]
    print("Starting analysis")
    for k in compounds:
        for n in n_neighb:
            for m in min_dist:
                filename = f"grit_umap_specs3k_{k}_{n}_{m}.png"
                full_path = os.path.join("Figures_SPECS3K", "umap_param_benchmark", filename)
                # Check if file exists
                if os.path.exists(full_path):
                    print(f"File {filename} already exists. Skipping to next.")
                    continue
                print(f"Now running compound {k} with {n} neighbors and min_dist = {m}")
                start_time = time.time()
                dat = dat.filter(pl.col("Metadata_cmpdName").is_in(["[DMSO]", k]))
                print(f"Data shape: {dat.shape}")
                if m == 0.01:
                    res = run_umap_and_merge(dat, features_fixed, n_neigh= n, min_dist = m, option = 'standard')
                else:
                    res = run_umap_and_merge(dat, features_fixed, n_neigh= n, min_dist = m, option = 'cuml')

                make_jointplot_seaborn_benchmark(res.to_pandas(), "Metadata_cmpdName",k, samp_meth = n, samp_rate = m)
                torch.cuda.empty_cache()

                end_time = time.time()  # Record end time of the iteration
                iteration_time = end_time - start_time 
                print(f"Analysis for {n}, {m} took {iteration_time:.2f} seconds")



def sample_groups(df, grouping_cols, ratio):
    subsampled_data = []
    # Group by the specified columns only in the filtered DataFrame
    grouped = df.groupby(grouping_cols)
    # For each group, subsample and append to the subsampled_data list, with progress bar
    for name, group in grouped:
        #if not (any(df["Metadata_cmpdName"].unique()) == "[DMSO]") & int(len(group)*ratio) < 5:
        #    subsampled_group = group
        #else:
        subsampled_group = group.sample(fraction=ratio, seed=42)
        subsampled_data.append(subsampled_group)
    # Concatenate the subsampled groups together
    subsampled_df = pl.concat(subsampled_data)
    return subsampled_df

def sample_compounds(df1, df2, sampling_rate, mode ="normal"):
    #Filter out sorbitol because neg control comp
    df1 = df1.filter(~(pl.col("Metadata_cmpdName") == "[SORB]"))
    df2 = df2.filter(~(pl.col("Metadata_cmpdName") == "[SORB]"))
    # Step 1: Extract "DMSO" values from first DataFrame

    # Sample DMSO group based on this ratio
    # Initialize dictionary to hold dataframes for each compound including DMSO
    if mode == "dmso_only":
        # Sample only DMSO group, ensuring random distribution from each well
        dmso_sampled  = sample_groups(df1.filter(pl.col("Metadata_cmpdName") == "[DMSO]"), ["Metadata_Plate", "Metadata_Well"], sampling_rate)
        dmso_sampled = dmso_sampled.with_columns(pl.lit("cell_na").alias("Metadata_Cell_Identity"))
        dmso_sampled = dmso_sampled.with_columns(pl.lit(0).cast(pl.Float64).alias("grit"))
        dmso_sampled = dmso_sampled.with_columns(dmso_sampled['moa'].cast(pl.Utf8))
        df2 = df2.with_columns(df2['moa'].cast(pl.Utf8))
        sampled_df = pl.concat([dmso_sampled, df2.filter(~((pl.col("Metadata_cmpdName") == "[DMSO]"))).drop("group")])
        return(sampled_df)
    # Sample other compounds
    elif mode == "normal":
        dmso_values = list(df2.filter(pl.col("Metadata_cmpdName") == "[DMSO]")["Metadata_Cell_Identity"])
        avg_compound_counts = df1.groupby(["Metadata_cmpdName", "Metadata_Plate", "Metadata_Well"]).agg(pl.count().alias('count')).group_by("Metadata_cmpdName").agg(pl.mean("count").alias("avg_count"))
        # Find the compound with the highest average count, excluding "DMSO"
        average_comp = avg_compound_counts.filter(~(pl.col("Metadata_cmpdName") == "[DMSO]")).select(pl.max("avg_count"))["avg_count"][0]
        # Calculate the average count for DMSO
        dmso_avg_count = avg_compound_counts.filter(pl.col("Metadata_cmpdName") == "[DMSO]")["avg_count"][0]
        # Calculate ratio for DMSO based on the compound with the highest average count
        dmso_ratio = (dmso_avg_count / average_comp)
        max_rows = 0
        cmpd_sampled = pl.DataFrame()
        for cmpd in df1["Metadata_cmpdName"].unique():
            if cmpd != "[DMSO]":
                cmpd_sample = sample_groups(df2.filter(pl.col("Metadata_cmpdName") == cmpd), ["Metadata_Plate", "Metadata_Well"], sampling_rate).drop("group")
                cmpd_sample = cmpd_sample.with_columns(cmpd_sample['moa'].cast(pl.Utf8))
                cmpd_sampled = pl.concat([cmpd_sampled, cmpd_sample])
                num_rows = cmpd_sample.shape[0]
                if num_rows > max_rows:
                    max_rows = num_rows

        if max_rows*dmso_ratio < len(dmso_values):
            # if dmso samples would be smaller than number of grit reference dmso cells, only sample using those!
            dmso_df = df2.filter(pl.col("Metadata_cmpdName") == "[DMSO]").drop("group")
        else:
            dmso_df = df1.filter(pl.col("Metadata_cmpdName") == "[DMSO]")

        dmso_sample = max_rows*dmso_ratio / dmso_df.shape[0]
        dmso_sampled  = sample_groups(dmso_df, ["Metadata_Plate", "Metadata_Well"], dmso_sample)
        dmso_sampled = dmso_sampled.with_columns(pl.lit("cell_na").alias("Metadata_Cell_Identity"))
        dmso_sampled = dmso_sampled.with_columns(pl.lit(0).cast(pl.Float64).alias("grit"))
        dmso_sampled = dmso_sampled.with_columns(dmso_sampled['moa'].cast(pl.Utf8))
        return pl.concat([dmso_sampled, cmpd_sampled])

    elif mode == "equal_sampling":
        cmpd_sampled = pl.DataFrame()
        for cmpd in df1["Metadata_cmpdName"].unique():
            if cmpd == "[DMSO]":
                cmpd_sample = sample_groups(df1.filter(pl.col("Metadata_cmpdName") == cmpd), ["Metadata_Plate", "Metadata_Well"], sampling_rate).drop("group")
                cmpd_sample = cmpd_sample.with_columns(pl.lit("cell_na").alias("Metadata_Cell_Identity"))
                cmpd_sample = cmpd_sample.with_columns(pl.lit(0).cast(pl.Float64).alias("grit"))
            else:
                cmpd_sample = sample_groups(df2.filter(pl.col("Metadata_cmpdName") == cmpd), ["Metadata_Plate", "Metadata_Well"], sampling_rate).drop("group")
            cmpd_sample = cmpd_sample.with_columns(cmpd_sample['moa'].cast(pl.Utf8))
            cmpd_sampled = pl.concat([cmpd_sampled, cmpd_sample])
        return cmpd_sampled    
    else:
        print(f"{mode} not valid as a sampling mode!")



def make_jointplot_seaborn_benchmark(embedding, colouring, cmpd, samp_meth, samp_rate, overlay=False, overlay_df=None):
    
    def get_color(val):
        if "[DMSO]" in val:
            return "lightgrey"
        else:
            return "#e96565"
    
    def get_size(val):
        return 20 if val != "[DMSO]" else 10
    
    embedding['color'] = embedding[colouring].apply(get_color)
    embedding['size'] = embedding[colouring].apply(get_size)

    all_treatments = list(embedding[colouring].unique())
    sorted_treatments = all_treatments.copy()
    specific_value = '[DMSO]'
    if specific_value in sorted_treatments:
        sorted_treatments.remove(specific_value)
    sorted_treatments.insert(0, specific_value)

    g = sns.JointGrid(x='UMAP1', y='UMAP2', data=embedding, height=10)

    for treatment in sorted_treatments:
        subset = embedding[embedding[colouring] == treatment]
        
        sns.kdeplot(x=subset["UMAP1"], ax=g.ax_marg_x, fill=True, color=get_color(treatment), legend=False)
        sns.kdeplot(y=subset["UMAP2"], ax=g.ax_marg_y, fill=True, color=get_color(treatment), legend=False)

    for treatment in sorted_treatments:
        subset = embedding[embedding[colouring] == treatment]
        g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], label=f"{treatment} - {len(subset)} cells", alpha=0.7, edgecolor='white', linewidth=0.5)

    # Overlay additional points if the option is active
    if overlay and overlay_df is not None:
        overlay_df['color'] = overlay_df[colouring].apply(get_color)
        # Increase the size for the overlay points
        overlay_df['size'] = overlay_df[colouring].apply(lambda val: get_size(val) * 2)  
        
        for treatment in sorted_treatments:
            subset = overlay_df[overlay_df[colouring] == treatment]
            g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], alpha=0.9, edgecolor='grey', linewidth=0.5)

    g.ax_joint.set_title(f"{cmpd} {samp_meth} rate: {samp_rate}")
    g.ax_joint.legend()
    filename = f"grit_umap_specs3k_{cmpd}_{samp_meth}_{samp_rate}.png"
    plt.savefig(os.path.join("Figures_SPECS3K", "umap_param_benchmark",filename), dpi=300, bbox_inches='tight')
    #plt.show()

def run_umap_and_merge(df, features, option = 'cuml', n_neigh = None, spread = 4, min_dist=0.1, n_components=2, metric='cosine', aggregate=False):
    # Filter the DataFrame for features and metadata
    feature_data = df.select(features).to_pandas()
    meta_features = [col for col in df.columns if col not in features]
    meta_data = df.select(meta_features)
    #n_neighbors = 100
    if n_neigh is None:
        n_neigh = math.ceil(np.sqrt(len(feature_data)))
    # Run UMAP with cuml
    print(f"Starting UMAP with {n_neigh} neighbors")
    if option == "cuml":
        umap_model = cuml.UMAP(n_neigh=n_neigh, spread= spread,  min_dist=min_dist, n_components=n_components, metric=metric).fit(feature_data)
        umap_embedding = umap_model.transform(feature_data)
    elif option == "standard":
        umap_model = umap.UMAP(n_neighbors=15, spread = spread, min_dist=min_dist, n_components=n_components, metric=metric, n_jobs = -1)
        umap_embedding = umap_model.fit_transform(feature_data)
    else:
        print(f"Option not available. Please choose 'cuml' or 'standard'")

    #cu_score = cuml.metrics.trustworthiness( feature_data, umap_embedding )
    #print(" cuml's trustworthiness score : ", cu_score )
    
    # Convert UMAP results to DataFrame and merge with metadata
    umap_df = pl.DataFrame(umap_embedding)

    old_column_name = umap_df.columns[0]
    old_column_name2 = umap_df.columns[1]
    # Rename the column
    new_column_name = "UMAP1"
    new_column_name2 = "UMAP2"
    umap_df = umap_df.rename({old_column_name: new_column_name, old_column_name2: new_column_name2})

    merged_df = pl.concat([meta_data, umap_df], how="horizontal")


    if aggregate:
        print("Aggregating data")
        aggregated_data = (df.groupby(['Metadata_Plate', 'Metadata_Well', 'Metadata_cmpdName']).agg([pl.col(feature).mean().alias(feature) for feature in features]))
        aggregated_data = aggregated_data.to_pandas()
        print(aggregated_data)
        aggregated_umap_embedding = umap_model.transform(aggregated_data[features])
        umap_agg = pl.DataFrame(aggregated_umap_embedding)
        umap_agg = umap_agg.rename({old_column_name: new_column_name, old_column_name2: new_column_name2})

        aggregated_meta_data = pl.DataFrame(aggregated_data[['Metadata_Plate', 'Metadata_Well', 'Metadata_cmpdName']])
        merged_agg = pl.concat([aggregated_meta_data, umap_agg], how="horizontal")
        return merged_df, merged_agg

    else:
        return merged_df
    

if __name__ == "__main__":
    main()