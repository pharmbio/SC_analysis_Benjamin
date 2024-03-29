import time
import tqdm
import polars as pl
import numpy as np
from sklearn.cluster import KMeans
import random
from sklearn_extra.cluster import KMedoids


"Most simple sampling function"
def subsample_dataset_pl(df, grouping_cols, fraction=0.5):
    '''
    Subsample a dataset while preserving the distribution of plates and Metadata_cmpdName using Polars.

    Parameters:
    - df: The original Polars DataFrame.
    - plate_column: The column name representing the plates.
    - cmpd_column: The column name representing the Metadata_cmpdName.
    - fraction: The fraction of data to keep for each group. Default is 0.5 (50%).

    Returns:
    - A subsampled Polars DataFrame.
    '''

    # Start tracking time
    start_time = time.time()

    # Initialize an empty list to store subsampled data from each group
    subsampled_data = []

    # Group by plates and Metadata_cmpdName
    grouped = df.groupby(grouping_cols)

    # For each group, subsample and append to the subsampled_data list, with progress bar
    for name, group in tqdm(grouped, desc="Subsampling groups", unit="group"):
        group_size = group.height
        subsample_size = int(group_size * fraction)
        subsampled_group = group.sample(n=subsample_size, seed=42)
        subsampled_data.append(subsampled_group)

    # Concatenate all subsampled groups together
    subsampled_df = pl.concat(subsampled_data)

    # Print running time
    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f} seconds.")

    return subsampled_df




"Sampling different strategies (well-ratio)"
def sample_groups(df, grouping_cols, ratio):
    subsampled_data = []
    # Group by the specified columns only in the filtered DataFrame
    grouped = df.groupby(grouping_cols)
    # For each group, subsample and append to the subsampled_data list, with progress bar
    for name, group in tqdm.tqdm(grouped, desc="Subsampling groups", unit="group"):
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




"Representative cell finding"

def find_representative_cells(df, group_column, feature_columns, method='random', n=1):
    # Ensure feature_columns is a list
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]

    # Initialize a list to hold the selected rows
    selected_rows = []

    # Group by the specified column
    groups = df.groupby(group_column)

    for name, group in tqdm.tqdm(groups):
        if name == "unassigned":
            print(name, "not a valid cluster")
            continue
        # Apply the selection method
        if method == 'random':
            # Randomly select n rows from the group
            selected_rows.append(group.sample(n=n))

        elif method == 'geomean':
            # Calculate the geometric mean of the feature columns for each group
            geomean = group[feature_columns].apply(lambda x: np.prod(x)**(1/len(x)), axis=0)
            # Find the row closest to the geometric mean
            closest = (group[feature_columns] - geomean).apply(np.linalg.norm, axis=1).idxmin()
            selected_rows.append(group.loc[[closest]])

        elif method == 'kmeans':
            if group.shape[0] > 60:
                n_cells_in_each_cluster_unif = 30
            else:
                n_cells_in_each_cluster_unif = int(group.shape[0] / 5)

            n_clusts = int(group.shape[0] / n_cells_in_each_cluster_unif)
            # Apply k-means clustering on the feature columns to find the most representative row
            kmeans = KMeans(n_clusters=n_clusts, random_state=0, n_init=10).fit(group[feature_columns])
            centroid = kmeans.cluster_centers_[0]
            closest = (group[feature_columns] - centroid).apply(np.linalg.norm, axis=1).idxmin()
            selected_rows.append(group.loc[[closest]])

        elif method == 'kmedoid':
            # Check if group is smaller than n
            if len(group) < n:
                raise ValueError(f"Group {name} has fewer rows than the number of requested representatives.")

            # Initialize and fit the KMedoids
            kmedoids = KMedoids(n_clusters=n, random_state=0).fit(group[feature_columns].values)
            
            # Get the indices of the medoids
            medoids_indices = kmedoids.medoid_indices_

            # Select rows corresponding to medoids
            for index in medoids_indices:
                selected_rows.append(group.iloc[[index]])

        else:
            raise ValueError("Unknown method: choose 'random', 'geomean', or 'kmeans'")

    # Concatenate all selected rows into a new DataFrame
    result_df = pd.concat(selected_rows, axis=0).reset_index(drop=True)
    sorted_df = result_df.sort_values(by=['Metadata_cmpdName'], ascending=[True])
    return sorted_df