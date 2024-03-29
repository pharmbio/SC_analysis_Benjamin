import os
import scipy.stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import pycytominer as pm
import anndata as ad
import scanpy as sc 
#from harmonypy import run_harmony
import polars as pl
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd 

def prep_data(df1, features, meta_features, plate):
    mask = ~df1['Metadata_cmpdName'].isin(['BLANK', 'UNTREATED', 'null'])
    filtered_df = df1[mask]
    temp1 = filtered_df.copy()
    temp1 = temp1.loc[(temp1['Metadata_Plate'] == plate)]
    data_norm = pm.normalize(profiles = temp1, features =  features, meta_features = meta_features, samples = "Metadata_cmpdName == '[DMSO]'", method = "mad_robustize")
    print("Feature selection starts, shape:", data_norm.shape)
    df_selected = pm.feature_select(data_norm, features = features, operation = ['correlation_threshold', 'drop_na_columns'], corr_threshold=0.8)
    print('Number of columns removed:', data_norm.shape[1] - df_selected.shape[1])
    removed_cols = set(data_norm.columns) - set(df_selected.columns)
    #out = df_selected.dropna().reset_index(drop = True)
    #print('Number of NA rows removed:', df_selected.shape[0] - out.shape[0])
    df_selected["Metadata_cmpdNameConc"] = df_selected["Metadata_cmpdName"] + df_selected["Metadata_cmpdConc"].astype(str)
    return data_norm, removed_cols


def find_file_with_string(directory, string):
    """
    Finds a file in the specified directory that contains the given string in its name.

    Args:
    directory (str): The directory to search in.
    string (str): The string to look for in the file names.

    Returns:
    str: The path to the first file found that contains the string. None if no such file is found.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return None

    # Iterate through all files in the directory
    for file in os.listdir(directory):
        if string in file:
            return os.path.join(directory, file)

    # Return None if no file is found
    return None


def create_anndata(df, aggregated):
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    data_columns = features  # Columns containing the feature data
    if aggregated:
        meta_data_columns = ['Plate','Well', 'Site', 'Metadata_cmpdName']
    else:
        meta_data_columns = meta_features

    X = df[data_columns].values
    obs = df[meta_data_columns]
    adata = ad.AnnData(X=X, obs=obs)
    return adata

def obsm_to_df(adata: ad.AnnData, obsm_key: str,
               columns: str) -> pd.DataFrame:
    '''Convert AnnData object to DataFrame using obs and obsm properties'''
    meta = adata.obs.reset_index(drop=True)
    feats = adata.obsm[obsm_key]
    n_feats = feats.shape[1]
    if not columns:
        columns = [f'{obsm_key}_{i:04d}' for i in range(n_feats)]
    data = pd.DataFrame(feats, columns=columns)
    return pd.concat([meta, data], axis=1)



def load_grit_data(folder, plates):
    """
    Processes Parquet files in the given folder based on whether their filenames contain 
    any of the strings in identifier_list. Merges 'Feature' and 'Metric' data based on 
    a specific column and concatenates with 'Control' data.

    :param folder_path: Path to the folder containing Parquet files.
    :param identifier_list: List of strings to be searched in the file names.
    :param merge_column: Column name on which to merge 'Feature' and 'Metric' data.
    :return: Combined Polars DataFrame.
    """
    feature_dfs = pl.DataFrame()
    metric_dfs = pl.DataFrame()

    # Iterate over files in the directory
    for plate in tqdm.tqdm(plates):
        file_names = [file for file in os.listdir(folder) if plate in file]
        if len(file_names) == 0:
            print(f"Plate {plate} not found")
            continue
        neg_path = [file for file in file_names if "neg_control" in file][0]
        neg_cells = pl.read_parquet(os.path.join(folder, neg_path))["Metadata_Cell_Identity"].unique()
        for i in file_names:
            file_path = os.path.join(folder, i)
            if "sc_features" in i:
                feat = pl.read_parquet(file_path).filter(((pl.col("Metadata_Cell_Identity").is_in(neg_cells))) |  ~(pl.col("Metadata_cmpdName") == "[DMSO]"))
                feature_dfs = pl.concat([feature_dfs, feat])
            elif "sc_grit" in i:
                metrics = pl.read_parquet(file_path)
                metrics_treat = metrics.filter(pl.col("group") == pl.col("comp")).drop("comp")
                metrics_ctrl = metrics.filter(pl.col("group") != pl.col("comp")).drop("comp")
                metric_dfs = pl.concat([metric_dfs, metrics_treat, metrics_ctrl])
                #metric_df = pl.read_parquet(i).drop("comp") if metric_df is None else metric_df.vstack(pl.read_parquet(i).drop("comp"))
    
    metric_df = metric_dfs.unique(subset=["Metadata_Cell_Identity"])
    # Merge Feature and Metric DataFrames
    merged_df = feature_dfs.join(metric_df, on="Metadata_Cell_Identity", how= "inner")
    # Concatenate Control DataFrames and merge with the above
    #final_df = pl.concat([merged_df, control_dfs])
    #.unique(subset = ["Metadata_Cell_Identity"])
    merged_df.write_parquet(os.path.join(folder, "sc_grit_FULL.parquet"))
    return merged_df


def merge_locations(df, location_folder):

    out_df = pl.DataFrame()
    combinations = df.unique(["Metadata_Plate", "Metadata_Well", "Metadata_Site"])
    # Iterate through unique combinations of Plate, Well, and Site
    for combination in tqdm.tqdm(combinations.to_pandas().itertuples(index=False), total = len(combinations)):
        plate, well, site = combination.Metadata_Plate, combination.Metadata_Well, combination.Metadata_Site

        # Construct the file path for the CSV
        file_path = f"{location_folder}/{plate}/{well}-{site}-Nuclei.csv"

        # Check if the file exists
        if os.path.exists(file_path):
            # Read the CSV file
            csv_df = pl.read_csv(file_path)
            filter = df.filter((pl.col("Metadata_Plate") == plate) &
                                            (pl.col("Metadata_Well") == well) &
                                            (pl.col("Metadata_Site") == site))
            # Ensure that csv_df aligns with the subset of original df in terms of row count
            if len(csv_df) != len(filter):
                # Handle error or misalignment
                print(f"{combination} doesn't match")  # or log it, or raise an error
            temp = pl.concat([filter, csv_df], how = "horizontal")
            out_df = pl.concat([out_df, temp], how = "vertical")
            # Perform the column concatenation operation
            # Assuming the order of rows in csv_df corresponds exactly to the order in the subset of df
            
    return out_df


def read_and_merge_single_file(df, plate, well, site, location_folder):
    file_path = f"{location_folder}/{plate}/{well}-{site}-Nuclei.csv"
    if os.path.exists(file_path):
        csv_df = pl.read_csv(file_path)
        filter_df = df.filter((pl.col("Metadata_Plate") == plate) &
                              (pl.col("Metadata_Well") == well) &
                              (pl.col("Metadata_Site") == site))
        if len(csv_df) == len(filter_df):
            return pl.concat([filter_df, csv_df], how="horizontal")
    return None

def merge_locations_parallel(df, location_folder, max_workers=10):
    combinations = df.unique(["Metadata_Plate", "Metadata_Well", "Metadata_Site"])
    dfs_to_concat = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create and submit tasks
        future_to_combination = {
            executor.submit(read_and_merge_single_file, df, comb["Metadata_Plate"], comb["Metadata_Well"], comb["Metadata_Site"], location_folder): comb 
            for comb in combinations.to_dicts()
        }
        
        for future in tqdm.tqdm(as_completed(future_to_combination), total=len(future_to_combination)):
            result = future.result()
            if result is not None:
                dfs_to_concat.append(result)
    
    # Concatenate all DataFrames at once at the end
    out_df = pl.concat(dfs_to_concat, how="vertical")
    return out_df

def run_scanpy(adata):
    sc.tl.pca(adata, svd_solver='arpack')
    #sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.pp.neighbors(adata)
    sc.tl.paga(adata, groups = "Metadata_cmpdName")
    sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    sc.tl.umap(adata, init_pos='paga')
    sc.tl.leiden(adata, key_added='clusters', resolution=0.2)

    return adata

def fix_keys(adata):
    def find_key_with_substring(obsm, substring):
        for key in obsm.keys():
            if substring in key:
                return key
        return None

    # Find the keys
    pca_key = find_key_with_substring(adata.obsm, 'pca')
    umap_key = find_key_with_substring(adata.obsm, 'dmso')
    if umap_key == None:
        umap_key = find_key_with_substring(adata.obsm, 'emb')

    # Rename the keys if they are found
    if pca_key:
        adata.obsm['X_pca'] = adata.obsm[pca_key]
        #del adata.obsm[pca_key]

    if umap_key:
        adata.obsm['X_umap'] = adata.obsm[umap_key]
        #del adata.obsm[umap_key]

    return adata 


def aggregate_by_group(adata, group_by):
    """
    Aggregate the expression data in an AnnData object by a specified group.
    
    Parameters:
    adata (AnnData): The original AnnData object.
    group_by (str): The column in adata.obs to group by.
    
    Returns:
    AnnData: A new AnnData object with aggregated data.
    """
    # Ensure the group_by column is categorical for efficiency
    adata.obs[group_by] = adata.obs[group_by].astype('category')
    if isinstance(adata.X, (np.ndarray, np.generic)):  # If .X is already a dense matrix
         adata_df = pd.DataFrame(adata.X, columns=aadata.var_names)
    else:  # If .X is a sparse matrix
        adata_df  = pd.DataFrame(adata.X.toarray(), columns=adata.var_names)

    # Group and aggregate data

    adata_df[group_by] = adata.obs[group_by].values
    
    # Aggregate data by taking the mean for each group
    aggregated_data = adata_df.groupby(group_by).median()
    # Create a new AnnData object with the aggregated data
    # Note: Here we're assuming that the .var information remains the same
    # If there are .obs specific fields you'd like to retain or calculate, adjust as needed
    aggregated_adata = anndata.AnnData(X=aggregated_data.values, var=adata.var.copy())
    aggregated_adata.obs[group_by] = aggregated_data.index.values
    
    return aggregated_adata