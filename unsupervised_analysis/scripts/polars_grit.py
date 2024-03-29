import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import polars as pl
import tqdm
import time
import pyarrow
from sklearn.preprocessing import StandardScaler
import numpy as np 
from typing import List
from collections import OrderedDict


def evaluate_grit(
    profiles: pl.DataFrame,
    features: List[str],
    meta_features: List[str],
    replicate_groups: dict,
    operation: str = "replicate_reproducibility",
    similarity_metric: str = "pearson",
    grit_control_perts: List[str] = ["None"],
    grit_replicate_summary_method: str = "mean",
):
    """Evaluate profile quality using the 'grit' metric in a Polars DataFrame.

    Parameters
    ----------
    profiles : pl.DataFrame
        Profiles must be a Polars DataFrame with profile samples as rows and profile
        features as columns. The columns should contain both metadata and feature
        measurements.
    features : list
        A list of strings corresponding to feature measurement column names in the
        `profiles` DataFrame. All features listed must be found in `profiles`.
    meta_features : list
        A list of strings corresponding to metadata column names in the `profiles`
        DataFrame. All features listed must be found in `profiles`.
    replicate_groups : dict
        A dict with keys "profile_col" and "replicate_group_col" indicating columns
        that store identifiers for each profile and higher order replicate information,
        respectively.
    similarity_metric: str, optional
        How to calculate pairwise similarity. Defaults to "pearson".
    grit_control_perts : list, optional
        Specific profile identifiers used as a reference when calculating grit.
    grit_replicate_summary_method : str, optional
        Defines how the replicate z scores are summarized, either "mean" or "median".

    Returns
    -------
    pl.DataFrame
        The resulting 'grit' metric as a Polars DataFrame.
    """
    if operation != "mp_value":
        # Melt the input profiles to long format
        similarity_melted_df = metric_melt(
            df=profiles,
            features=features,
            metadata_features=meta_features,
            similarity_metric=similarity_metric,
            eval_metric=operation,
        )
    metric_result = grit_pl(
            similarity_melted_df=similarity_melted_df,
            control_perts=grit_control_perts,
            profile_col=replicate_groups["profile_col"],
            replicate_group_col=replicate_groups["replicate_group_col"],
            replicate_summary_method=grit_replicate_summary_method,
        )
    return metric_result

def grit_pl(
    similarity_melted_df: pl.DataFrame,
    control_perts: List[str],
    profile_col: str,
    replicate_group_col: str,
    replicate_summary_method: str = "mean",
) -> pl.DataFrame:

    # Determine pairwise replicates
    similarity_melted_df = assign_replicates_pl(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=[profile_col, replicate_group_col],
    )

    # Check to make sure that the melted dataframe is full
    assert_melt(similarity_melted_df, eval_metric="grit")

    # Extract out specific columns
    pair_ids = set_pair_ids()
    profile_col_name = "{x}{suf}".format(
        x=profile_col, suf=pair_ids[list(pair_ids)[0]]["suffix"]
    )
    # Define the columns to use in the calculation
    column_id_info = set_grit_column_info(
        profile_col=profile_col, replicate_group_col=replicate_group_col
    )
    results = []
    # Calculate grit for each perturbation
    for value in similarity_melted_df[profile_col_name].unique():
        group_df = similarity_melted_df.filter(pl.col(profile_col_name) == value)
        print(group_df)
        result_df = calculate_grit_pl(
            replicate_group_df=group_df,
            control_perts=control_perts,
            column_id_info=column_id_info,
            replicate_summary_method=replicate_summary_method
        )
        results.append(result_df)

# Concatenate all results into a single DataFrame
    grit_df = pl.concat(results)

    return grit_df

def assert_melt(
    df: pl.DataFrame, eval_metric: str = "replicate_reproducibility"
) -> None:
    r"""Helper function to ensure that we properly melted the pairwise correlation
    matrix

    Downstream functions depend on how we process the pairwise correlation matrix. The
    processing is different depending on the evaluation metric.

    Parameters
    ----------
    df : pandas.DataFrame
        A melted pairwise correlation matrix
    eval_metric : str
        The user input eval metric

    Returns
    -------
    None
        Assertion will fail if we incorrectly melted the matrix
    """
    df = df.to_pandas()
    pair_ids = set_pair_ids()
    df = df.loc[:, [pair_ids[x]["index"] for x in pair_ids]]
    index_sums = df.sum().tolist()

    assert_error = "Stop! The eval_metric provided in 'metric_melt()' is incorrect!"
    assert_error = "{err} This is a fatal error providing incorrect results".format(
        err=assert_error
    )
    if eval_metric == "replicate_reproducibility":
        assert index_sums[0] != index_sums[1], assert_error
    elif eval_metric == "precision_recall":
        assert index_sums[0] == index_sums[1], assert_error
    elif eval_metric == "grit":
        assert index_sums[0] == index_sums[1], assert_error
    elif eval_metric == "hitk":
        assert index_sums[0] == index_sums[1], assert_error



def calculate_grit_pl(
    replicate_group_df: pd.DataFrame,
    control_perts: List[str],
    column_id_info: dict,
    distribution_compare_method: str = "zscore",
    replicate_summary_method: str = "mean",
):
    """
    Calculate grit using Polars.

    Parameters
    ----------
    replicate_group_df : pl.DataFrame
        A DataFrame storing pairwise correlations of all profiles to a single replicate group.
    control_perts : list
        The profile_ids that should be considered controls (the reference).
    column_id_info : dict
        A dictionary of column identifiers noting profile and replicate group ids.
    distribution_compare_method : str, optional
        How to compare the replicate and reference distributions of pairwise similarity.
    replicate_summary_method : str, optional
        How to summarize replicate z-scores. Defaults to "mean".

    Returns
    -------
    pl.DataFrame
        A DataFrame with grit score per perturbation.
    """

    # We must have helper functions like check_compare_distribution_method and 
    # check_replicate_summary_method implemented for Polars or validated beforehand.
    replicate_group_df = pl.DataFrame(replicate_group_df)
    group_entry = get_grit_entry(replicate_group_df, column_id_info["group"]["id"])
    pert = get_grit_entry(replicate_group_df, column_id_info["profile"]["id"])
    # Define distributions for control perturbations
    control_distrib = replicate_group_df.filter(
        pl.col(column_id_info["profile"]["comparison"]).is_in(control_perts)
    )["similarity_metric"].to_numpy().reshape(-1, 1)
    assert control_distrib.shape[0] > 1, "Error! No control perturbations found."

    # Define distributions for the same group (but not the same perturbation)
    same_group_distrib = replicate_group_df.filter(
        (pl.col(column_id_info["group"]["comparison"]) == group_entry) &
        (pl.col(column_id_info["profile"]["comparison"]) != pert)
    )["similarity_metric"].to_numpy().reshape(-1, 1)

    # Compute the grit score
    # Assuming compare_distributions is a function that can operate on numpy arrays
    # and return a single grit score.
    if same_group_distrib.size == 0:
        grit_score = None
    else:
        grit_score = compare_distributions(
            target_distrib=same_group_distrib,
            control_distrib=control_distrib,
            method=distribution_compare_method,
            replicate_summary_method=replicate_summary_method,
        )

    return_bundle = {
        "perturbation": [pert],
        "group": [group_entry],
        "grit": [grit_score if grit_score is not None else float('nan')]
    }
    print(return_bundle)
    return pd.series(return_bundle)

def assign_replicates_pl(
    similarity_melted_df: pl.DataFrame,
    replicate_groups: List[str]
) -> pl.DataFrame:
    """Determine which profiles should be considered replicates in a Polars DataFrame.

    Parameters
    ----------
    similarity_melted_df : pl.DataFrame
        Long Polars DataFrame of annotated pairwise correlations.
    replicate_groups : list
        a list of metadata column names in the original profile dataframe used to
        indicate replicate profiles.

    Returns
    -------
    pl.DataFrame
        Updated similarity_melted_df with additional columns indicating replicate comparisons.
    """
    print(similarity_melted_df.columns)
    pair_ids = set_pair_ids()
    replicate_col_names = {x: "{x}_replicate".format(x=x) for x in replicate_groups}
    #print(pair_ids)
    compare_dfs_exprs = []
    for replicate_col in replicate_groups:
        replicate_cols_with_suffix = [
            pl.col(f"{replicate_col}{pair_ids[x]['suffix']}")
            for x in pair_ids
        ]
        print(f"{replicate_col}{pair_ids['pair_b']['suffix']}")
        # Check if all replicate columns are present
        assert all(
            [f"{replicate_col}{pair_ids[x]['suffix']}" in similarity_melted_df.columns for x in pair_ids]
        ), "replicate_group not found in melted dataframe columns"

        replicate_col_name = replicate_col_names[replicate_col]
        
        # Create expressions for each comparison
        compare_dfs_exprs.extend([
            pl.when(replicate_cols_with_suffix[0] == replicate_cols_with_suffix[1])
             .then(True)
             .otherwise(False)
             .alias(replicate_col_name)
        ])

    # Apply all expressions and create a group_replicate column
    group_replicate_expr = pl.fold(
        acc=True,
        f=lambda acc, x: acc & x,
        exprs=[pl.col(name) for name in replicate_col_names.values()]
    ).alias('group_replicate')

    def min_rowwise(*columns):
        return pl.reduce(lambda a, b: pl.when(a < b).then(a).otherwise(b), columns)

    # Calculate row-wise minimum by dynamically unpacking the replicate columns
    group_replicate_expr = min_rowwise(*[pl.col(name) for name in replicate_col_names.values()])

    # Add the new group_replicate column
    compare_df = compare_df.with_column(group_replicate_expr.alias("group_replicate"))

    # Select the columns of interest (replicate_col_names and group_replicate)
    compare_df = compare_df.select(
        list(replicate_col_names.values()) + ["group_replicate"]
    )
    compare_dfs_exprs.append(group_replicate_expr)

    # Add the comparison columns to the original DataFrame
    similarity_melted_df = similarity_melted_df.with_columns(compare_dfs_exprs)

    return similarity_melted_df


def metric_melt(
    df: pl.DataFrame,
    features: List[str],
    metadata_features: List[str],
    eval_metric: str = "replicate_reproducibility",
    similarity_metric: str = "pearson",
) -> pl.DataFrame:
    # Make sure all features and metadata features are present
    assert all([x in df.columns for x in metadata_features]), "Metadata feature not found"
    assert all([x in df.columns for x in features]), "Profile feature not found"

    # Subset DataFrame to specific features and metadata features
    meta_df = df.select(metadata_features)
    feature_df = df.select(features)

    # Get pairwise metric matrix
    # Assuming get_pairwise_metric_pl is a rewritten version of get_pairwise_metric for Polars
    pair_df = get_pairwise_metric(df=feature_df.to_pandas(), similarity_metric=similarity_metric)
    
    # Convert pairwise matrix into metadata-labeled melted matrix
    # Assuming process_melt_pl is a rewritten version of process_melt for Polars
    output_df = process_melt(df=pair_df, meta_df=meta_df, eval_metric=eval_metric)

    return output_df

def get_pairwise_metric(df: pd.DataFrame, similarity_metric: str = 'pearson') -> pd.DataFrame:
    """Calculate the pairwise similarity metric for a feature-only DataFrame using numpy.

    Parameters
    ----------
    df : pd.DataFrame
        Samples x features, where all columns can be coerced to floats
    similarity_metric : str, optional
        The pairwise comparison to calculate, defaults to 'pearson'.

    Returns
    -------
    pd.DataFrame
        A pairwise similarity matrix
    """
    assert similarity_metric in ["pearson"], "Unsupported similarity metric"

    # Convert the DataFrame to numpy array and transpose it
    data = df.to_numpy().T

    # Calculate correlation using numpy
    if similarity_metric == 'pearson':
        corr_matrix = np.corrcoef(data)
    else:
        NotImplementedError("Only pearson correlation valid")

    corr_df = pd.DataFrame(corr_matrix)
    # Convert the numpy array back to pandas DataFrame
    return pl.DataFrame(corr_df)



def get_grit_entry_pl(df: pl.DataFrame, col: str) -> str:
    """
    Helper function to define the perturbation identifier of interest for Polars DataFrame.

    Grit must be calculated using unique perturbations. This may or may not mean unique
    perturbations in the DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame containing the data.
    col : str
        The name of the column from which to extract the perturbation identifier.

    Returns
    -------
    str
        The unique entry from the specified column.

    Raises
    ------
    AssertionError
        If there is not exactly one unique value in the column.
    """
    entries = df.select(col).unique().to_series()
    assert entries.len() == 1, "grit is calculated for each perturbation independently"
    return entries[0]


def process_melt_pl(
    df: pl.DataFrame,
    meta_df: pl.DataFrame,
    eval_metric: str = "replicate_reproducibility",
) -> pl.DataFrame:
    assert df.height == df.width, "Matrix must be symmetrical"

    pair_ids = set_pair_ids()  # Assuming this function is properly adapted for Polars

    if eval_metric == "replicate_reproducibility":
        upper_tri = get_upper_matrix(df)
        df = df.mask(upper_tri)
    else:
        df2 = df.to_pandas()
        np.fill_diagonal(df2.values, np.nan)
        df3 = pl.DataFrame(df2)
    df_with_index = df3.with_row_count("index")
    # Melt the DataFrame
    metric_unlabeled_df = df_with_index.melt(id_vars="index", value_vars=df.columns, variable_name=pair_ids["pair_b"]["index"], value_name="similarity_metric")

# Dropping NA values
    metric_unlabeled_df = metric_unlabeled_df.filter(pl.col("similarity_metric").is_not_null())

    # Renaming the column 'index' to pair_ids["pair_a"]["index"]
    metric_unlabeled_df = metric_unlabeled_df.rename({"index": pair_ids["pair_a"]["index"]})

    # Merge metadata
    if "index" not in meta_df.columns:
        meta_df = meta_df.with_row_count("index")
        meta_df = meta_df.with_columns(meta_df["index"].cast(pl.Utf8))

    metric_unlabeled_df = metric_unlabeled_df.with_columns(metric_unlabeled_df[pair_ids["pair_b"]["index"]].cast(pl.Utf8))

    # First Merge
    # Joining metric_unlabeled_df with meta_df

    for col in metric_unlabeled_df.columns:
        if col in meta_df.columns and col not in [pair_ids["pair_b"]["index"], "index"]:
            metric_unlabeled_df = metric_unlabeled_df.rename({col: col + pair_ids["pair_b"]["suffix"]})

    first_merge = metric_unlabeled_df.join(
        meta_df,
        left_on=pair_ids["pair_b"]["index"],
        right_on="index"
    )

    first_merge = first_merge.with_columns(first_merge[pair_ids["pair_a"]["index"]].cast(pl.Utf8))
    # Second Merge
    # Joining the result of the first merge with meta_df again
    for col in first_merge.columns:
        if col in meta_df.columns and col not in [pair_ids["pair_a"]["index"], "index"]:
            first_merge = first_merge.rename({col: col + pair_ids["pair_a"]["suffix"]})

    output_df = first_merge.join(
        meta_df,
        left_on=pair_ids["pair_a"]["index"],
        right_on="index")
    
    return output_df

def process_melt(
    df: pl.DataFrame,
    meta_df: pl.DataFrame,
    eval_metric: str = "replicate_reproducibility",
) -> pl.DataFrame:
    """Helper function to annotate and process an input similarity matrix

    Parameters
    ----------
    df : pandas.DataFrame
        A similarity matrix output from
        :py:func:`cytominer_eval.transform.transform.get_pairwise_metric`
    meta_df : pandas.DataFrame
        A wide matrix of metadata information where the index aligns to the similarity
        matrix index
    eval_metric : str, optional
        Which metric to ultimately calculate. Determines whether or not to keep the full
        similarity matrix or only one diagonal. Defaults to "replicate_reproducibility".

    Returns
    -------
    pandas.DataFrame
        A pairwise similarity matrix
    """
    # Confirm that the user formed the input arguments properly

    df = df.to_pandas()
    meta_df = meta_df.to_pandas()
    assert df.shape[0] == df.shape[1], "Matrix must be symmetrical"


    # Get identifiers for pairing metadata
    pair_ids = set_pair_ids()

    # Subset the pairwise similarity metric depending on the eval metric given:
    #   "replicate_reproducibility" - requires only the upper triangle of a symmetric matrix
    #   "precision_recall" - requires the full symmetric matrix (no diagonal)
    # Remove pairwise matrix diagonal and redundant pairwise comparisons
    if eval_metric == "replicate_reproducibility":
        upper_tri = get_upper_matrix(df)
        df = df.where(upper_tri)
    else:
        np.fill_diagonal(df.values, np.nan)
    # Convert pairwise matrix to melted (long) version based on index value
    metric_unlabeled_df = (
        pd.melt(
            df.reset_index(),
            id_vars="index",
            value_vars=df.columns,
            var_name=pair_ids["pair_b"]["index"],
            value_name="similarity_metric",
        )
        .dropna()
        .reset_index(drop=True)
        .rename({"index": pair_ids["pair_a"]["index"]}, axis="columns")
    )

    # Merge metadata on index for both comparison pairs
    metric_unlabeled_df['pair_b_index'] = metric_unlabeled_df['pair_b_index'].astype('int64')
    
    output_df = meta_df.merge(
        meta_df.merge(
            metric_unlabeled_df,
            left_index=True,
            right_on=pair_ids["pair_b"]["index"],
        ),
        left_index=True,
        right_on=pair_ids["pair_a"]["index"],
        suffixes=[pair_ids["pair_a"]["suffix"], pair_ids["pair_b"]["suffix"]],
    ).reset_index(drop=True)

    output_df2= pl.DataFrame(output_df)
    return output_df2


def get_upper_matrix(df: pl.DataFrame, batch_size = 10000) -> np.ndarray:
    """
    Generate an upper triangle mask for a large DataFrame in batches.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame for which the upper triangle mask is to be created.
    batch_size : int
        The size of each batch.

    Returns
    -------
    np.ndarray
        An upper triangle matrix the same shape as the input DataFrame.
    """
    nrows, ncols = df.shape
    upper_matrix = np.zeros((nrows, ncols), dtype=bool)
    
    for start_row in range(0, nrows, batch_size):
        end_row = min(start_row + batch_size, nrows)
        for start_col in range(0, ncols, batch_size):
            end_col = min(start_col + batch_size, ncols)
            
            # Create a mask for the current batch
            batch_mask = np.triu(np.ones((end_row - start_row, end_col - start_col)), k=1).astype(bool)
            
            # Place the batch mask in the corresponding position of the full matrix
            upper_matrix[start_row:end_row, start_col:end_col] = batch_mask

    return upper_matrix



def set_pair_ids():
    r"""Helper function to ensure consistent melted pairiwise column names

    Returns
    -------
    collections.OrderedDict
        A length two dictionary of suffixes and indeces of two pairs.
    """
    pair_a = "pair_a"
    pair_b = "pair_b"

    return_dict = OrderedDict()
    return_dict[pair_a] = {
        "index": "{pair_a}_index".format(pair_a=pair_a),
        "suffix": "_{pair_a}".format(pair_a=pair_a),
    }
    return_dict[pair_b] = {
        "index": "{pair_b}_index".format(pair_b=pair_b),
        "suffix": "_{pair_b}".format(pair_b=pair_b),
    }

    return return_dict


def set_grit_column_info(profile_col: str, replicate_group_col: str) -> dict:
    """Transform column names to be used in calculating grit

    In calculating grit, the data must have a metadata feature describing the core
    replicate perturbation (profile_col) and a separate metadata feature(s) describing
    the larger group (replicate_group_col) that the perturbation belongs to (e.g. gene,
    MOA).

    Parameters
    ----------
    profile_col : str
        the metadata column storing profile ids. The column can have unique or replicate
        identifiers.
    replicate_group_col : str
        the metadata column indicating a higher order structure (group) than the
        profile column. E.g. target gene vs. guide in a CRISPR experiment.

    Returns
    -------
    dict
        A nested dictionary of renamed columns indicating how to determine replicates
    """
    # Identify column transform names
    pair_ids = set_pair_ids()

    profile_id_with_suffix = [
        "{col}{suf}".format(col=profile_col, suf=pair_ids[x]["suffix"])
        for x in pair_ids
    ]

    group_id_with_suffix = [
        "{col}{suf}".format(col=replicate_group_col, suf=pair_ids[x]["suffix"])
        for x in pair_ids
    ]

    col_info = ["id", "comparison"]
    profile_id_info = dict(zip(col_info, profile_id_with_suffix))
    group_id_info = dict(zip(col_info, group_id_with_suffix))

    column_id_info = {"profile": profile_id_info, "group": group_id_info}
    return column_id_info

def compare_distributions(
    target_distrib: List[float],
    control_distrib: List[float],
    method: str = "zscore",
    replicate_summary_method: str = "mean",
) -> float:
    """Compare two distributions and output a single score indicating the difference.

    Given two different vectors of distributions and a comparison method, determine how
    the two distributions are different.

    Parameters
    ----------
    target_distrib : np.array
        A list-like (e.g. numpy.array) of floats representing the first distribution.
        Must be of shape (n_samples, 1).
    control_distrib : np.array
        A list-like (e.g. numpy.array) of floats representing the second distribution.
        Must be of shape (n_samples, 1).
    method : str, optional
        A string indicating how to compare the two distributions. Defaults to "zscore".
    replicate_summary_method : str, optional
        A string indicating how to summarize the resulting scores, if applicable. Only
        in use when method="zscore".

    Returns
    -------
    float
        A single value comparing the two distributions
    """
    # Confirm that we support the provided methods

    if method == "zscore":
        scaler = StandardScaler()
        scaler.fit(control_distrib)
        scores = scaler.transform(target_distrib)

        if replicate_summary_method == "mean":
            scores = np.mean(scores)
        elif replicate_summary_method == "median":
            scores = np.median(scores)

    return scores


def get_grit_entry(df: pl.DataFrame, col: str) -> str:
    """Helper function to define the perturbation identifier of interest

    Grit must be calculated using unique perturbations. This may or may not mean unique
    perturbations.
    """
    entries = df[col].unique()
    assert len(entries) == 1, "grit is calculated for each perturbation independently"
    return str(entries[0])
