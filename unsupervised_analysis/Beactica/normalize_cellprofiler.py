import tqdm
import polars as pl
import scipy.stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import pycytominer as pm
import os
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.base import BaseEstimator, TransformerMixin


def main():
    PROJECT_ROOT = '/share/data/analyses/benjamin/Single_cell_project_rapids/SPECS3K'
    feat_out = "cellprofiler/feature_parquets/"
    meta_features = ['Metadata_Plate',
                    'Metadata_cmpdName',
                    'Metadata_Well',
                    'Metadata_Site',
                    'Location_Center_X_nuclei',
                    'Location_Center_Y_nuclei',
                    'ImageNumber_nuclei',
                    'ObjectNumber_nuclei',
                    'Metadata_AcqID_nuclei',
                    'FileName_CONC_nuclei',
                    'FileName_HOECHST_nuclei',
                    'FileName_ICF_CONC_nuclei',
                    'FileName_ICF_HOECHST_nuclei',
                    'FileName_ICF_MITO_nuclei',
                    'FileName_ICF_PHAandWGA_nuclei',
                    'FileName_ICF_SYTO_nuclei',
                    'FileName_MITO_nuclei',
                    'FileName_PHAandWGA_nuclei',
                    'FileName_SYTO_nuclei',
                    'PathName_CONC_nuclei',
                    'PathName_HOECHST_nuclei',
                    'PathName_ICF_CONC_nuclei',
                    'PathName_ICF_HOECHST_nuclei',
                    'PathName_ICF_MITO_nuclei',
                    'PathName_ICF_PHAandWGA_nuclei',
                    'PathName_ICF_SYTO_nuclei',
                    'PathName_MITO_nuclei',
                    'PathName_PHAandWGA_nuclei',
                    'PathName_SYTO_nuclei']
    

    cols_to_drop = ['Children_cytoplasm_Count_nuclei',
 'Location_Center_Z_nuclei',
 'Neighbors_FirstClosestObjectNumber_Adjacent_nuclei',
 'Neighbors_SecondClosestObjectNumber_Adjacent_nuclei',
 'Number_Object_Number_nuclei',
 'Parent_cells_nuclei',
 'ImageNumber_cells',
 'Metadata_AcqID_cells',
 'FileName_CONC_cells',
 'FileName_HOECHST_cells',
 'FileName_ICF_CONC_cells',
 'FileName_ICF_HOECHST_cells',
 'FileName_ICF_MITO_cells',
 'FileName_ICF_PHAandWGA_cells',
 'FileName_ICF_SYTO_cells',
 'FileName_MITO_cells',
 'FileName_PHAandWGA_cells',
 'FileName_SYTO_cells',
 'PathName_CONC_cells',
 'PathName_HOECHST_cells',
 'PathName_ICF_CONC_cells',
 'PathName_ICF_HOECHST_cells',
 'PathName_ICF_MITO_cells',
 'PathName_ICF_PHAandWGA_cells',
 'PathName_ICF_SYTO_cells',
 'PathName_MITO_cells',
 'PathName_PHAandWGA_cells',
 'PathName_SYTO_cells',
 'Children_cytoplasm_Count_cells',
 'Children_nuclei_Count_cells',
 'Location_Center_Z_cells',
 'Neighbors_FirstClosestObjectNumber_Adjacent_cells',
 'Neighbors_SecondClosestObjectNumber_Adjacent_cells',
 'Number_Object_Number_cells',
 'Parent_precells_cells',
 'ImageNumber_cytoplasm',
 'Metadata_AcqID_cytoplasm',
 'FileName_CONC_cytoplasm',
 'FileName_HOECHST_cytoplasm',
 'FileName_ICF_CONC_cytoplasm',
 'FileName_ICF_HOECHST_cytoplasm',
 'FileName_ICF_MITO_cytoplasm',
 'FileName_ICF_PHAandWGA_cytoplasm',
 'FileName_ICF_SYTO_cytoplasm',
 'FileName_MITO_cytoplasm',
 'FileName_PHAandWGA_cytoplasm',
 'FileName_SYTO_cytoplasm',
 'PathName_CONC_cytoplasm',
 'PathName_HOECHST_cytoplasm',
 'PathName_ICF_CONC_cytoplasm',
 'PathName_ICF_HOECHST_cytoplasm',
 'PathName_ICF_MITO_cytoplasm',
 'PathName_ICF_PHAandWGA_cytoplasm',
 'PathName_ICF_SYTO_cytoplasm',
 'PathName_MITO_cytoplasm',
 'PathName_PHAandWGA_cytoplasm',
 'PathName_SYTO_cytoplasm',
 'Number_Object_Number_cytoplasm',
 'Parent_cells_cytoplasm',
 'Parent_nuclei_cytoplasm']
    
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

    for p in tqdm.tqdm(plates):
        # Construct the file path using a function that finds the correct file
        file_path = find_file_with_string(os.path.join(PROJECT_ROOT, "cellprofiler/feature_parquets/"), f"sc_profiles_cellprofiler_{p}")
        if file_path is not None:
            try:
                print(f"Reading in plate {p}")
                feature_df = pl.read_parquet(file_path)
                feature_df = feature_df.drop(cols_to_drop)
                feature_df = feature_df.with_columns(
                                            [
                                                pl.col(column).cast(pl.Float32)
                                                for column in feature_df.columns
                                                if column not in meta_features and feature_df[column].dtype == pl.Float64
                                            ]
                                        )
                features = [col for col in feature_df.columns if col not in meta_features]
                #Run normalization
                print(f"Starting normalization on plate {p}")
                temp_processed = prep_data(feature_df, features, meta_features, p)
                print(f"Plate {p} normalized. Writing to disk.")
                temp_processed.write_parquet(os.path.join("/share/data/analyses/benjamin/Single_cell_project_rapids/SPECS3K/cellprofiler/feature_parquets/", f'sc_profiles_normalized_cellprofiler_{p}.parquet'))
                #temp_polars = pl.from_pandas(temp_processed)
                #mad_norm_df = pl.concat([mad_norm_df, temp_polars])

            except Exception as e:
                print(f"Error processing plate {p}: {e}")

    #cols_to_drop = list(set().union(*drop_cols.values()))
    #mad_norm_df = mad_norm_df.drop(cols_to_drop)
    #print("Writing normalized features to disk")
    #mad_norm_df.write_parquet(os.path.join("Beactica/Results", 'sc_profiles_normalized_Beactica.parquet'))


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
    return print(f"No file found with {string}")





def prep_data(df1, features, meta_features, plate):
    #mask = ~df1['compound_id'].isin(['BLANK', 'UNTREATED', 'null'])
    df1_filt = df1.filter(df1["AreaShape_Area_cells"].is_not_null())
    print(df1.shape[0] - df1_filt.shape[0], "na rows dropped")
    data_norm = normalize_pl(profiles = df1_filt, features =  features, meta_features = meta_features, samples = "Metadata_cmpdName", value = "[DMSO]", method = "standardize")
    #print("Feature selection starts, shape:", data_norm.shape)
    #df_selected = pm.feature_select(data_norm, features = features, operation = ["blocklist", 'correlation_threshold', 'drop_na_columns'], corr_threshold=0.8,)
    #print('Number of columns removed:', data_norm.shape[1] - df_selected.shape[1])
    #removed_cols = set(data_norm.columns) - set(df_selected.columns)
    #out = df_selected.dropna().reset_index(drop = True)
    #print('Number of NA rows removed:', df_selected.shape[0] - out.shape[0])
    #df_selected["Metadata_cmpdNameConc"] = df_selected["Metadata_cmpdName"] + df_selected["Metadata_cmpdConc"].astype(str)
    return data_norm

def normalize_pl(
    profiles,
    features,
    meta_features,
    samples="all",
    value = "DMSO",
    method="standardize",
    output_file=None,
    output_type="csv",
    compression_options=None,
    float_format=None,
    mad_robustize_epsilon=1e-18,
):

    # Define which scaler to use
    method = method.lower()
    avail_methods = ["standardize", "robustize", "mad_robustize"]
    if method not in avail_methods:
        raise ValueError("operation must be one of {}".format(avail_methods))

    # Selecting the scaler
    if method == "standardize":
        scaler = StandardScaler()
    elif method == "robustize":
        scaler = RobustScaler()
    elif method == "mad_robustize":
        scaler = RobustMAD(epsilon=mad_robustize_epsilon)

    # Extract feature and meta data as Numpy arrays for scaling
    feature_data = profiles.select(features).to_pandas()
    meta_df = profiles.select(meta_features)

    # Fit and transform
    if samples == "all":
        fitted_scaler = scaler.fit(feature_data)
    else:
        # For subsetting, convert to Pandas for query functionality, then back to Numpy
        subset_data = profiles.filter(pl.col(samples) == value).select(features).to_pandas()
        fitted_scaler = scaler.fit(subset_data)
    scaled_features = fitted_scaler.transform(feature_data)

    # Construct new Polars DataFrame for the output
    scaled_feature_df = pl.DataFrame({features[i]: scaled_features[:, i] for i in range(len(features))})

    normalized = meta_df.hstack(scaled_feature_df)

    return normalized

def load_profiles(profiles):
    """
    Unless a dataframe is provided, load the given profile dataframe from path or string

    Parameters
    ----------
    profiles : {str, pathlib.Path, pandas.DataFrame}
        file location or actual pandas dataframe of profiles

    Return
    ------
    pandas DataFrame of profiles

    Raises:
    -------
    FileNotFoundError
        Raised if the provided profile does not exists
    """
    if not isinstance(profiles, pl.DataFrame):
        # Check if path exists and load depending on file type
        print("No polars give, please provide correct input format!")
    return profiles

    
class RobustMAD(BaseEstimator, TransformerMixin):
    """Class to perform a "Robust" normalization with respect to median and mad

        scaled = (x - median) / mad

    Attributes
    ----------
    epsilon : float
        fudge factor parameter
    """

    def __init__(self, epsilon=1e-18):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Compute the median and mad to be used for later scaling.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            dataframe to fit RobustMAD transform

        Returns
        -------
        self
            With computed median and mad attributes
        """
        # Get the mean of the features (columns) and center if specified
        self.median = X.median()
        # The scale param is required to preserve previous behavior. More info at:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_absolute_deviation.html#scipy.stats.median_absolute_deviation
        self.mad = pd.Series(
            median_abs_deviation(X, nan_policy="omit", scale=1 / 1.4826),
            index=self.median.index,
        )
        return self

    def transform(self, X, copy=None):
        """Apply the RobustMAD calculation

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            dataframe to fit RobustMAD transform

        Returns
        -------
        pandas.core.frame.DataFrame
            RobustMAD transformed dataframe
        """
        return (X - self.median) / (self.mad + self.epsilon)


if __name__ == "__main__":
    main()
