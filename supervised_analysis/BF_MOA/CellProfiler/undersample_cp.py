import polars as pl
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import tqdm
import gc
from imblearn.under_sampling import NearMiss
import numpy as np
from sklearn.preprocessing import LabelEncoder

def main():
    specs2k_plates = ['P103617',
                    'P103602',
                    'P103595',
                    'P103597',
                    'P103613',
                    'P103591',
                    'P103615',
                    'P103607',
                    'P103619',
                    'P103606',
                    'P103616',
                    'P103601',
                    'P103603',
                    'P103620',
                    'P103614',
                    'P103621',
                    'P103593',
                    'P103592',
                    'P103612',
                    'P103608',
                    'P103600',
                    'P103609',
                    'P103618',
                    'P103589',
                    'P103605',
                    'P103590',
                    'P103599',
                    'P103610',
                    'P103604',
                    'P103611',
                    'P103598',
                    'P103596',
                    'P103594']
    specs3k_plates = ['P101382',
                    'P101339',
                    'P101338',
                    'P101337',
                    'P101354',
                    'P101350',
                    'P101360',
                    'P101375',
                    'P101363',
                    'P101335',
                    'P101373',
                    'P101372',
                    'P101352',
                    'P101334',
                    'P101369',
                    'P101336',
                    'P101345',
                    'P101377',
                    'P101346',
                    'P101366',
                    'P101359',
                    'P101361',
                    'P101364',
                    'P101365',
                    'P101362',
                    'P101374',
                    'P101380',
                    'P101367',
                    'P101358',
                    'P101342',
                    'P101371',
                    'P101341',
                    'P101368',
                    'P101348',
                    'P101370',
                    'P101379',
                    'P101386',
                    'P101353',
                    'P101381',
                    'P101351',
                    'P101357',
                    'P101384',
                    'P101347',
                    'P101343',
                    'P101387',
                    'P101385',
                    'P101355',
                    'P101340',
                    'P101378',
                    'P101344',
                    'P101349',
                    'P101376',
                    'P101356']
    specs5k_sc_features_total = pl.read_parquet("datasets/sc_profiles_classification_specs5k_total.parquet")
    specs5k_sc_features_total = specs5k_sc_features_total.filter(~pl.col("Metadata_Plate").is_in(["P103620", "P103621", "P103619", "P101387", "P101386", "P101385", "P101384"])) #Filter out DMSO plates
    specs5k_sc_features_total = encode_labels(specs5k_sc_features_total)
    treatment_plates = specs5k_sc_features_total.filter(pl.col('moa') != 'dmso').select('Metadata_Plate').unique()

    # Step 2: Filter rows to keep only those with "DMSO" in treatment and plate values from the list
    filtered_df = specs5k_sc_features_total.filter((pl.col('moa') != 'dmso') | (pl.col('moa') == 'dmso') & (pl.col('Metadata_Plate').is_in(treatment_plates['Metadata_Plate'])))
    print(f"{specs5k_sc_features_total.shape} before filter, {filtered_df.shape} after filtering.")
    gc.collect()
    print("Sampling")
    resampled_specs5k_big = undersampling(filtered_df, "control_group_sampling")
    gc.collect()
    resampled_specs5k_big = prepare_class_data(resampled_specs5k_big, specs2k_plates, specs3k_plates)
    gc.collect()
    print("Save to file")
    resampled_specs5k_big.write_parquet("datasets/specs5k_undersampled_moa_CP_BF.parquet")



def encode_labels(df):
    le = LabelEncoder()
    le.fit(df["moa"])
    df_labels = list(le.transform(df["moa"])) 
    df = df.with_columns(pl.Series(name="label", values=df_labels))  
    return df 


def undersampling(df, strategy):
    df_pd = df.to_pandas()
    if strategy == "nearmmiss":
        feature_cols = [col for col in df.columns if "Feature" in col]
        metadata_cols = [col for col in df.columns if col not in feature_cols]
        metadata_cols.remove("label")
        nm = NearMiss(version=1, n_jobs= -1)

        # Split features and target
        #X = specs3k_sc_features_pandas[[col for col in specs3k_sc_features_total.columns if not "label"]]
        X = df_pd[feature_cols]
        y = df_pd['label']

        # Apply NearMiss
        X_res, y_res = nm.fit_resample(X, y)

        df_resampled = pl.DataFrame(X_res)
        df_resampled = df_resampled.with_columns(pl.Series('label', y_res))

        resampled_df = df_resampled.join(df, on = feature_cols, how='left')
        resampled_df = resampled_df.drop("")
    elif strategy == "control_group_sampling":
        # Identify the most abundant class and its size
        
        # Assuming 'control_label' is the label of your control group
        #control_label = 6
        control_label = df.filter(pl.col("moa") =="dmso").select("label").unique()[0]
        print(control_label)
        
        # Filter the DataFrame for the control group and other groups
        control_group = df.filter(pl.col('label') == control_label)
        other_groups = df.filter(pl.col('label') != control_label)

        value_counts = other_groups.select(pl.col('label')).group_by('label').agg(pl.count().alias('count'))
        most_abundant_class_size = value_counts.select(pl.max('count')).to_numpy()[0][0]

        sample_rate = most_abundant_class_size/(control_group.shape[0])
        print(sample_rate)
        
        if 0.1 < sample_rate < 1.0:
            # Randomly sample rows to achieve approximately the target size
            control_grouped = (control_group.group_by(["Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_cmpdName"]))
            sampled = control_grouped.apply(lambda x: x.sample(fraction=sample_rate, seed = 42))
        elif sample_rate < 0.1:
            #control_grouped = (control_group.group_by(["Metadata_Plate", "Metadata_Well", "Metadata_cmpdName"]))
            #sampled = control_grouped.apply(lambda x: x.sample(fraction=sample_rate, seed = 42))
            groups = control_group.partition_by(["Metadata_Plate", "Metadata_Well", "Metadata_cmpdName"])

            # Sample from each group
            sampled_partitions = [group.sample(fraction=sample_rate, seed=42) for group in groups]

            # Concatenate the sampled partitions back into a single DataFrame
            sampled = pl.concat(sampled_partitions)
            print(f"{sampled.shape} DMSO cells sampled")
                    
        # Concatenate the sampled control group back with the other data
        resampled_df = pl.concat([other_groups, sampled])
    
    return resampled_df


def prepare_class_data(df, plate2k, plate3k):
    df = df.drop('')
    df = df.with_columns(
    pl.when(pl.col('Metadata_Plate').is_in(plate2k)).then(pl.lit("specs2k"))
    .when(pl.col('Metadata_Plate').is_in(plate3k)).then(pl.lit("specs3k")) 
    .otherwise(pl.lit("other"))
    .alias('project')
    )
    return df

if __name__ == '__main__':
    main()
