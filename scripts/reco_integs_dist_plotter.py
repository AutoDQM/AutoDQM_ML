'''
Macro to use post-scripts/assess.py to convert the lsit of SSE scores to ROC curves for studies over a large data set
Requires input directory where bad_runs_sse_scores.csv is located (this is also the output directory) and list of bad
runs as labelled by data certification reports or similar (i.e. not algos!) (Runs not in the list of bad runs are 
considered good runs by default)
'''

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import sys
import json
import argparse
import awkward

from autodqm_ml.utils import expand_path
from autodqm_ml.constants import kANOMALOUS, kGOOD
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--input_pca",
    help = "input file with PCA reco integrals (i.e. output from train.py)",
    type = str,
    required = True,
    default = None
  )
  parser.add_argument(
    "--input_ae",
    help = "input file with AE reco integrals (i.e. output from train.py)",
    type = str,
    required = True,
    default = None
  )
  return parser.parse_args()

def main(args):

  pca_df = pd.read_csv(args.input_pca)
  ae_df = pd.read_csv(args.input_ae)
  algorithm_name1 = str(pca_df['algo'].iloc[0]).upper()
  algorithm_name2 = str(ae_df['algo'].iloc[0]).upper()

  pca_df = pca_df.loc[:,~pca_df.columns.duplicated()].copy()
  ae_df = ae_df.loc[:,~ae_df.columns.duplicated()].copy()

  pca_df = pca_df.rename(columns={col: col + "_" + algorithm_name1 for col in pca_df.columns if "_reco_integ" in col})
  ae_df = ae_df.rename(columns={col: col + "_" + algorithm_name2 for col in ae_df.columns if "_reco_integ" in col})

  pca_cols = [col for col in pca_df.columns if '_reco_integrals_PCA' in col]
  ae_cols = [col for col in ae_df.columns if '_reco_integrals_AE' in col]

  pca_dict = {each_hist: "max" for each_hist in pca_cols}
  ae_dict = {each_hist: "max" for each_hist in ae_cols}

  pca_df = pca_df.groupby(['run_number','label'])[pca_cols].agg(pca_dict).reset_index()
  pca_df = pca_df.sort_values(['label']).reset_index()
  pca_df = pca_df[['run_number','label'] + [col for col in pca_df.columns if (col != 'run_number')&(col != 'label')]]

  ae_df = ae_df.groupby(['run_number','label'])[ae_cols].agg(ae_dict).reset_index()
  ae_df = ae_df.sort_values(['label']).reset_index()
  ae_df = ae_df[['run_number','label'] + [col for col in ae_df.columns if (col != 'run_number')&(col != 'label')]]

  pca_df_good = pca_df.loc[pca_df['label'] == 0].reset_index()
  pca_df_bad = pca_df.loc[pca_df['label'] == 1].reset_index()
  pca_df_good = pca_df_good[pca_cols]
  pca_df_bad = pca_df_bad[pca_cols]

  ae_df_good = ae_df.loc[ae_df['label'] == 0].reset_index()
  ae_df_bad = ae_df.loc[ae_df['label'] == 1].reset_index()
  ae_df_good = ae_df_good[ae_cols]
  ae_df_bad = ae_df_bad[ae_cols]

  print("[LOGGER] Successfully read files and divided into good and bad runs ready for plotting.")

  largest_integ_pca = max(pca_df_good.max().max(),pca_df_bad.max().max())
  largest_integ_ae = max(ae_df_good.max().max(),ae_df_bad.max().max())

  for i in range(pca_df_good.shape[1]): 
    
    ith_column_heading = pca_df_good.columns[i]
    hist_path = ith_column_heading.rsplit("/", 2)

    hist_name = "_".join(hist_path[-2:])[:-35]

    bin_width = 0.001

    #largest_integ_pca = max(pca_df_good.iloc[:, i].max(),pca_df_bad.iloc[:, i].max())
    #largest_integ_ae = max(ae_df_good.iloc[:, i].max(),ae_df_bad.iloc[:, i].max())

    bins_pca = np.arange(0.0,largest_integ_pca+bin_width, bin_width)
    bins_ae = np.arange(0.0,largest_integ_ae+bin_width, bin_width)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(hist_name)

    ax1.hist(pca_df_good.iloc[:, i], bins=bins_pca, color='green', edgecolor='green', alpha=0.5, label="Good runs")
    ax1.hist(pca_df_bad.iloc[:, i], bins=bins_pca, color='red', edgecolor='red', alpha=0.5, label="Bad runs")
    ax1.set_xlabel(r'PCA reconstructed integral values')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='upper right')

    ax2.hist(ae_df_good.iloc[:, i], bins=bins_ae, color='green', edgecolor='green', alpha=0.5, label="Good runs")
    ax2.hist(ae_df_bad.iloc[:, i], bins=bins_ae, color='red', edgecolor='red', alpha=0.5, label="Bad runs")
    ax2.set_xlabel(r'AE reconstructed integral values')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='upper right')
    


    plt.tight_layout()
    file_name = f'{hist_name}_reco_integ_dist.pdf'
    plt.savefig("reco_integ_dist_dir/"+file_name)
    
  print("[LOGGER] Finished plotting all "+str(pca_df_good.shape[1])+" histograms, saved to the 'reco_integ_dist_dir/' directory.")

if __name__ == "__main__":
  args = parse_arguments()
  main(args)
