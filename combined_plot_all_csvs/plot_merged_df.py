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

import json
import argparse
import awkward

from autodqm_ml.utils import expand_path
from autodqm_ml.constants import kANOMALOUS, kGOOD

def count_number_of_hists_above_threshold(Fdf, Fthreshold_list):
  runs_list = Fdf['run_number']
  Ft_list = np.array([float(Fthreshold_list_item) for Fthreshold_list_item in Fthreshold_list])
  hist_bad_count = 0
  bad_hist_array = []
  for run in runs_list:
    run_row = Fdf.loc[Fdf['run_number'] == run].drop(columns=['run_number'])
    run_row = run_row.iloc[0].values
    hist_bad_count = sum(hist_sse >= hist_thresh for hist_sse, hist_thresh in zip(run_row, Ft_list))
    bad_hist_array.append(hist_bad_count)
  print(len(bad_hist_array))
  return bad_hist_array

# returns mean number of runs with SSE above the given threshold
def count_mean_runs_above(Fdf, Fthreshold_list):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  mean_hists_flagged_per_run = sum(hists_flagged_per_run) / len(Fdf['run_number'])
  return mean_hists_flagged_per_run

# returns fraction of runs with SSE above the given threshold
def count_fraction_runs_above(Fdf, Fthreshold_list, N_bad_hists):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  count = len([i for i in hists_flagged_per_run if i >= N_bad_hists])
  count_per_run = count / len(Fdf['run_number'])
  return count_per_run

def main():

  sse_df = pd.read_csv("HLTPhysics_8_REF.csv")

  sse_df = sse_df.loc[:,~sse_df.columns.duplicated()].copy()
  
  hist_cols = [col for col in sse_df.columns if 'Run summary' in col]
  hist_dict = {each_hist: "max" for each_hist in hist_cols}
  #print(hist_dict)

  sse_df = sse_df.groupby(['run_number','label'])[hist_cols].agg(hist_dict).reset_index()

  sse_df = sse_df.sort_values(['label']).reset_index()
  sse_df = sse_df[['run_number','label'] + [col for col in sse_df.columns if (col != 'run_number')&(col != 'label')]]

  print(len(sse_df.columns))
  print(sse_df.columns)

  sse_df_good = sse_df.loc[(sse_df['label'] == 0) | (sse_df['label'] == -1)].reset_index()
  sse_df_bad = sse_df.loc[sse_df['label'] == 1].reset_index()
  sse_df_good = sse_df_good[['run_number'] + hist_cols]
  sse_df_bad = sse_df_bad[['run_number'] + hist_cols]

  # new threshold cut-offs per Si's recommendations
  # 0th cut-off at 1st highest SSE + (1st - 2nd highest)*0.5
  # 1st cut-off at mean<1st, 2nd> highest SSE
  # Nth cut-off at mean<Nth, N+1th> highest SSE
  cutoffs_across_hists = []
  for histogram in hist_cols:
    sse_ordered = sorted(sse_df_good[histogram], reverse=True)
    cutoff_0 = sse_ordered[0] + 0.5*(sse_ordered[0] - sse_ordered[1])
    cutoff_thresholds = []
    cutoff_thresholds.append(cutoff_0)
    for ii in range(len(sse_ordered)-1):
      cutoff_ii = 0.5*(sse_ordered[ii]+sse_ordered[ii+1])
      cutoff_thresholds.append(cutoff_ii)
    cutoffs_across_hists.append(cutoff_thresholds)

  cutoffs_across_hists = np.array(cutoffs_across_hists)

  N_bad_hists = [3]
  tFRF_ROC_good_X = []
  tFRF_ROC_bad_Y = []

  for nbh_ii in N_bad_hists:
    tFRF_ROC_good_X_init = []
    tFRF_ROC_bad_Y_init = []
    print(len(cutoffs_across_hists[0,:]))
    for cutoff_index in range(len(cutoffs_across_hists[0,:])):
      t_cutoff_index_g_FRF_rc = count_fraction_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index], nbh_ii)
      t_cutoff_index_b_FRF_rc = count_fraction_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index], nbh_ii)
      tFRF_ROC_good_X_init.append(t_cutoff_index_g_FRF_rc)
      tFRF_ROC_bad_Y_init.append(t_cutoff_index_b_FRF_rc)

    tFRF_ROC_good_X_init = sorted(tFRF_ROC_good_X_init)
    tFRF_ROC_bad_Y_init = sorted(tFRF_ROC_bad_Y_init)
    print(tFRF_ROC_bad_Y_init)

    tFRF_ROC_good_X.append(tFRF_ROC_good_X_init)
    tFRF_ROC_bad_Y.append(tFRF_ROC_bad_Y_init)

  tMHF_ROC_good_X = []
  tMHF_ROC_bad_Y = []
  for cutoff_index in range(len(cutoffs_across_hists[0,:])):
    #if not cutoff_index % 8:
    t_cutoff_index_g_MHF_rc = count_mean_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index])
    t_cutoff_index_b_MHF_rc = count_mean_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index])
    tMHF_ROC_good_X.append(t_cutoff_index_g_MHF_rc)
    tMHF_ROC_bad_Y.append(t_cutoff_index_b_MHF_rc)

  tMHF_ROC_good_X = sorted(tMHF_ROC_good_X)
  tMHF_ROC_bad_Y = sorted(tMHF_ROC_bad_Y)

  fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(12,6))

  axs[1].set_xlabel('Fraction of good runs with at least N histogram flags')
  axs[1].set_ylabel('Fraction of bad runs with at least N histogram flags')
  #print(N_bad_hists[jj])
  #print(tFRF_ROC_good_X[jj])
  #print(tFRF_ROC_bad_Y[jj])

  #print("hf_roc_good hf_roc_bad rf_n1_roc_good rf_n1_roc_bad rf_n3_roc_good rf_n3_roc_bad rf_n5_roc_good rf_n5_roc_bad")
  #for i in range(len(tMHF_ROC_good_X)):
  #  print(tMHF_ROC_good_X[i],tMHF_ROC_bad_Y[i],tFRF_ROC_good_X[2][i],tFRF_ROC_bad_Y[2][i],tFRF_ROC_good_X[1][i],tFRF_ROC_bad_Y[1][i],tFRF_ROC_good_X[0][i],tFRF_ROC_bad_Y[0][i])

  axs[1].plot(tFRF_ROC_good_X[0],tFRF_ROC_bad_Y[0], '-rD', mfc='purple', mec='k', markersize=8, linewidth=1, label='Flag thresholds, N = ' + str(N_bad_hists[0]))
  #axs[1].plot(tFRF_ROC_good_X[1],tFRF_ROC_bad_Y[1], '-bo', mfc='yellow', mec='k', markersize=8, linewidth=1, label='Flag thresholds, N = ' + str(N_bad_hists[1]))
  #axs[1].plot(tFRF_ROC_good_X[2],tFRF_ROC_bad_Y[2], '-g^', mfc='orange', mec='k', markersize=8, linewidth=1, label='Flag thresholds, N = ' + str(N_bad_hists[2]))
  axs[1].axis(xmin=0,xmax=0.35,ymin=0,ymax=0.8)
  axs[1].axline((0, 0), slope=1, linestyle='--',linewidth=0.8,color='gray')
  axs[1].annotate("Combined RF ROC", xy=(0.05, 0.95), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=12, weight='bold')
  axs[1].legend(loc='lower right')

  axs[0].set_xlabel('Mean number of histogram flags per good run')
  axs[0].set_ylabel('Mean number of histogram flags per bad run')
  axs[0].plot(tMHF_ROC_good_X,tMHF_ROC_bad_Y, '-rD', mfc='purple', mec='k', markersize=8, linewidth=1, label='Flag thresholds')
  axs[0].axline((0, 0), slope=1, linestyle='--',linewidth=0.8,color='gray')
  axs[0].annotate("Combined HF ROC", xy=(0.05, 0.95), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=12, weight='bold')
  axs[0].axis(xmin=0,xmax=8,ymin=0,ymax=25)
  axs[0].legend(loc='lower right')

  plt.savefig("./simple_ROC.pdf",bbox_inches='tight')
  print("SAVED: ./simple_ROC.pdf")

if __name__ == "__main__":
  main()
