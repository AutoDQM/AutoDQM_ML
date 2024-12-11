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
    "--input_sse",
    help = "input file with SSE scores (i.e. output from train.py)",
    type = str,
    required = True,
    default = None
  )
  parser.add_argument(
    "--input_chi2_maxpull",
    help = "input file with Chi2 and Max pull values (i.e. output from train.py)",
    type = str,
    required = True,
    default = None
  )
  return parser.parse_args()

def main(args):

  sse_df = pd.read_csv(args.input_sse)
  val_df = pd.read_csv(args.input_chi2_maxpull)
  algorithm_name = str(sse_df['algo'].iloc[0]).upper()

  if algorithm_name == "BETAB": algorithm_name = "Beta_Binomial"

  sse_df = sse_df.loc[:,~sse_df.columns.duplicated()].copy()
  val_df = val_df.loc[:,~val_df.columns.duplicated()].copy()
  sse_cols = [col for col in sse_df.columns if '_score_' in col]
  #chi2_cols = [col for col in val_df.columns if '_chi2_tol1' in col]
  chi2_cols = [col for col in val_df.columns if '_chi2prime' in col]
  mp_cols = [col for col in val_df.columns if '_chi2prime' in col]

  sse_dict = {each_hist: "max" for each_hist in sse_cols}
  chi2_dict = {each_hist: "max" for each_hist in chi2_cols}
  mp_dict = {each_hist: "max" for each_hist in mp_cols}

  sse_df = sse_df.groupby(['run_number','label'])[sse_cols].agg(sse_dict).reset_index()
  sse_df = sse_df.sort_values(['label']).reset_index()
  sse_df = sse_df[['run_number','label'] + [col for col in sse_df.columns if (col != 'run_number')&(col != 'label')]]

  chi2_df = val_df.groupby(['run_number','label'])[chi2_cols].agg(chi2_dict).reset_index()
  chi2_df = chi2_df.sort_values(['label']).reset_index()
  chi2_df = chi2_df[['run_number','label'] + [col for col in chi2_df.columns if (col != 'run_number')&(col != 'label')]]

  mp_df = val_df.groupby(['run_number','label'])[mp_cols].agg(mp_dict).reset_index()
  mp_df = mp_df.sort_values(['label']).reset_index()
  mp_df = mp_df[['run_number','label'] + [col for col in mp_df.columns if (col != 'run_number')&(col != 'label')]]

  sse_df_good = sse_df.loc[sse_df['label'] == 0].reset_index()
  sse_df_bad = sse_df.loc[sse_df['label'] == 1].reset_index()
  sse_df_good = sse_df_good[sse_cols]
  sse_df_bad = sse_df_bad[sse_cols]

  chi2_df_good = chi2_df.loc[chi2_df['label'] == 0].reset_index()
  chi2_df_bad = chi2_df.loc[chi2_df['label'] == 1].reset_index()
  chi2_df_good = chi2_df_good[chi2_cols]
  chi2_df_bad = chi2_df_bad[chi2_cols]

  mp_df_good = mp_df.loc[mp_df['label'] == 0].reset_index()
  mp_df_bad = mp_df.loc[mp_df['label'] == 1].reset_index()
  mp_df_good = mp_df_good[mp_cols]
  mp_df_bad = mp_df_bad[mp_cols]

  sse_df_good_log10 = np.log10(sse_df_good)
  sse_df_good_log10 = sse_df_good_log10.clip(lower=-6, upper=2)
  sse_df_bad_log10 = np.log10(sse_df_bad)
  sse_df_bad_log10 = sse_df_bad_log10.clip(lower=-6, upper=2)

  chi2_df_good_log2 = np.log2(chi2_df_good)
  chi2_df_good_log2 = chi2_df_good_log2.clip(lower=-5, upper=10)
  chi2_df_bad_log2 = np.log2(chi2_df_bad)
  chi2_df_bad_log2 = chi2_df_bad_log2.clip(lower=-5, upper=10)

  mp_df_good_log2 = np.log2(mp_df_good)
  mp_df_good_log2 = mp_df_good_log2.clip(lower=-5, upper=10)
  mp_df_bad_log2 = np.log2(mp_df_bad)
  mp_df_bad_log2 = mp_df_bad_log2.clip(lower=-5, upper=10)

  largest_sse = max(sse_df_good_log10.max().max(),sse_df_bad_log10.max().max())
  smallest_sse = min(sse_df_good_log10.min().min(),sse_df_bad_log10.min().min())
  largest_chi2 = max(chi2_df_good_log2.max().max(),chi2_df_bad_log2.max().max())
  smallest_chi2 = min(chi2_df_good_log2.min().min(),chi2_df_bad_log2.min().min())
  largest_mp = max(chi2_df_good_log2.max().max(),mp_df_bad_log2.max().max())
  smallest_mp = min(chi2_df_good_log2.min().min(),mp_df_bad_log2.min().min())


  print("[LOGGER] Successfully read files and divided into good and bad runs ready for plotting.")

  directory_path = 'raw_scores_dist_dir'
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created.")
  else:
    print(f"Directory '{directory_path}' already exists.")

  for i in range(sse_df_good.shape[1]): 
    
    ith_column_heading = sse_df_good.columns[i]
    hist_path = ith_column_heading.rsplit("/", 2)

    hist_name = "_".join(hist_path[-2:])[:-17]

    bin_width_scores = 0.05
    bin_width_vals = 0.1

    bins_sse = np.arange(smallest_sse-bin_width_scores,largest_sse+bin_width_scores, bin_width_scores)
    bins_chi2 = np.arange(smallest_chi2-bin_width_vals,largest_chi2+bin_width_vals, bin_width_vals)
    bins_mp = np.arange(smallest_mp-bin_width_vals,largest_mp+bin_width_vals, bin_width_vals)

    #bins_sse = np.arange(-6-bin_width_scores,0+bin_width_scores, bin_width_scores)
    #bins_chi2 = np.arange(-5-bin_width_vals,10+bin_width_vals, bin_width_vals)
    #bins_mp = np.arange(-5-bin_width_vals,10+bin_width_vals, bin_width_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(hist_name)

    ax1.hist(sse_df_good_log10.iloc[:, i], bins=bins_sse, color='green', edgecolor='green', alpha=0.5, label="Good runs")
    ax1.hist(sse_df_bad_log10.iloc[:, i], bins=bins_sse, color='red', edgecolor='red', alpha=0.5, label="Bad runs")
    ax1.set_xlabel(r'SSE score (log$_{10}$)')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='upper right')
    
    ax2.hist(chi2_df_good_log2.iloc[:, i], bins=bins_chi2, color='green', edgecolor='green', alpha=0.5, label="Good runs")
    ax2.hist(chi2_df_bad_log2.iloc[:, i], bins=bins_chi2, color='red', edgecolor='red', alpha=0.5, label="Bad runs")
    ax2.set_xlabel(r'Modified $\chi^{2}$ (log$_{2}$)')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='upper right')

    #ax3.hist(mp_df_good_log2.iloc[:, i], bins=bins_mp, color='green', edgecolor='green', alpha=0.5, label="Good runs")
    #ax3.hist(mp_df_bad_log2.iloc[:, i], bins=bins_mp, color='red', edgecolor='red', alpha=0.5, label="Bad runs")
    #ax3.set_xlabel(r'Max pull (log$_{2}$)')
    #ax3.set_ylabel('Frequency')
    #ax3.legend(loc='upper right')

    plt.tight_layout()
    file_name = f'{hist_name}_raw_scores_dist.pdf'
    plt.savefig("raw_scores_dist_dir/"+file_name)
    
  print("[LOGGER] Finished plotting all "+str(sse_df_good.shape[1])+" histograms, saved to the 'raw_scores_dist_dir/' directory.")

if __name__ == "__main__":
  args = parse_arguments()
  main(args)
