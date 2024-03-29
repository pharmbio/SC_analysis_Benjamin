import os
from pathlib import Path
import polars as pl

def find_highest_numbered_subfolder_with_file(root_folder, target_file = 'featICF_nuclei.parquet'):
    """
    Navigates through subfolders named as integers under the given root_folder.
    Returns the path of the file in the highest numbered subfolder that contains it.
    If the file isn't found in any subfolders, returns None.

    Parameters:
    root_folder (str): Path to the root folder containing numbered subfolders.
    target_file (str): Name of the file to search for in subfolders.
    """
    highest_file_path = None
    highest_number = -1

    for subdir, dirs, files in os.walk(root_folder):
        for dirname in dirs:
            # Attempt to convert folder name to an integer
            try:
                folder_number = int(dirname)
                # Check if this folder contains the target file
                potential_path = Path(subdir) / dirname / target_file
                if potential_path.exists() and folder_number > highest_number:
                    # Update highest number and file path if this is the largest so far
                    highest_number = folder_number
                    highest_file_path = potential_path
            except ValueError:
                # Non-integer folder names are ignored
                continue

    return highest_file_path



def add_suffix_to_column_names(df, suffix):
    """
    Adds a prefix and underscore to all column names in the Polars DataFrame.

    Parameters:
    df (pl.DataFrame): The original Polars DataFrame.
    prefix (str): The prefix string to add to each column name.

    Returns:
    pl.DataFrame: A new DataFrame with updated column names.
    """
    # Create a dictionary mapping old names to new names
    rename_dict = {col: f"{col}_{suffix}" for col in df.columns}

    # Rename the columns
    df = df.rename(rename_dict)

    return df


def load_and_stack_dataframes(df_list):
    """
    Loads multiple DataFrames, ensures column data types match, and stacks them.

    Parameters:
    df_list (list): A list of DataFrames to be stacked.

    Returns:
    pl.DataFrame: A new DataFrame with all provided DataFrames stacked.
    """

    # Initialize an empty list to hold the aligned DataFrames
    aligned_dfs = []

    # Define the target data types based on the first DataFrame as a reference
    # This assumes all DataFrames have the same column names and order
    reference_dtypes = df_list[0].dtypes

    for df in df_list:
        # Check each column's data type and cast if necessary
        for col, ref_dtype in zip(df.columns, reference_dtypes):
            if df[col].dtype != ref_dtype:
                df = df.with_columns(df[col].cast(ref_dtype))
        aligned_dfs.append(df)

    # Stack all the aligned DataFrames
    stacked_df = pl.concat(aligned_dfs)

    return stacked_df



def load_cellprofiler(meta):
    plates = ['P101334',
                'P101335',
                'P101336',
                'P101337',
                'P101338',
                'P101339',
                'P101340',
                'P101341',
                'P101342',
                'P101343',
                'P101344',
                'P101345',
                'P101346',
                'P101347',
                'P101348',
                'P101349',
                'P101350',
                'P101351',
                'P101352',
                'P101353',
                'P101354',
                'P101355',
                'P101356',
                'P101357',
                'P101358',
                'P101359',
                'P101360',
                'P101361',
                'P101362',
                'P101363',
                'P101364',
                'P101365',
                'P101366',
                'P101367',
                'P101368',
                'P101369',
                'P101370',
                'P101371',
                'P101372',
                'P101373',
                'P101374',
                'P101375',
                'P101376',
                'P101377',
                'P101378',
                'P101379',
                'P101380',
                'P101381',
                'P101382']
    #out_df = []
    for p in tqdm.tqdm(plates):
        print("Importing plate:", p)
        nuclei_feats = pl.read_parquet(find_highest_numbered_subfolder_with_file(os.path.join(CELLPROFILER_ROOT, p)))
        cyto_feats = pl.read_parquet(find_highest_numbered_subfolder_with_file(os.path.join(CELLPROFILER_ROOT, p), target_file= "featICF_cytoplasm.parquet"))
        cell_feats = pl.read_parquet(find_highest_numbered_subfolder_with_file(os.path.join(CELLPROFILER_ROOT, p), target_file= "featICF_cells.parquet"))
        nuclei_feats = add_suffix_to_column_names(nuclei_feats, "nuclei")
        cyto_feats = add_suffix_to_column_names(cyto_feats, "cytoplasm")
        cell_feats = add_suffix_to_column_names(cell_feats, "cells")   

        df = nuclei_feats.join(
        cell_feats,
        left_on=['Metadata_Barcode_nuclei', 'Metadata_Site_nuclei', 'Metadata_Well_nuclei','Parent_cells_nuclei'],
        right_on=[ 'Metadata_Barcode_cells','Metadata_Site_cells', 'Metadata_Well_cells','ObjectNumber_cells'],
        how='left'
        )
        df = df.join(
        cyto_feats, 
        left_on = ['Metadata_Barcode_nuclei','Metadata_Site_nuclei', 'Metadata_Well_nuclei','Parent_cells_nuclei'],
        right_on = ['Metadata_Barcode_cytoplasm','Metadata_Site_cytoplasm', 'Metadata_Well_cytoplasm','ObjectNumber_cytoplasm'], 
        how='left')

        df = df.with_columns(df["Location_Center_X_nuclei"].cast(pl.Int64))
        df = df.with_columns(df["Location_Center_Y_nuclei"].cast(pl.Int64))
        df = df.with_columns((pl.lit("s") + df["Metadata_Site_nuclei"].cast(pl.Utf8)).alias("Metadata_Site_nuclei"))
        
        #df.write_parquet(os.path.join(PROJECT_PATH, "cellprofiler/feature_parquets", f"sc_profiles_cellprofiler_{p}.parquet"))
        #temp = cell_locations.join(df, left_on = ["Metadata_Plate", "Metadata_Site", "Metadata_Well", "Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"], right_on=["Metadata_Barcode_nuclei", "Metadata_Site_nuclei", "Metadata_Well_nuclei", "Location_Center_X_nuclei", "Location_Center_Y_nuclei"], how = "inner")
        temp = df.join(meta.select(["Metadata_Plate", "Metadata_Site", "Metadata_cmpdName", "Metadata_Well"]).unique(), left_on = ["Metadata_Barcode_nuclei", "Metadata_Site_nuclei", "Metadata_Well_nuclei"], right_on = ["Metadata_Plate", "Metadata_Site", "Metadata_Well"], how = "left")
        temp = temp.rename({"Metadata_Barcode_nuclei": "Metadata_Plate", 
                     "Metadata_Well_nuclei": "Metadata_Well", 
                     "Metadata_Site_nuclei": "Metadata_Site"})
        temp.write_parquet(os.path.join(PROJECT_PATH, "cellprofiler/feature_parquets", f"sc_profiles_cellprofiler_{p}.parquet"))
        #out_df.append(temp)