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
    hist_bad_count = sum(hist_sse > hist_thresh for hist_sse, hist_thresh in zip(run_row, Ft_list))
    bad_hist_array.append(hist_bad_count)
  return bad_hist_array

# returns mean number of runs with SSE above the given threshold
def count_mean_runs_above(Fdf, Fthreshold_list):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  mean_hists_flagged_per_run = sum(hists_flagged_per_run) / len(Fdf['run_number'])
  return mean_hists_flagged_per_run

# returns fraction of runs with SSE above the given threshold
def count_fraction_runs_above(Fdf, Fthreshold_list, N_bad_hists):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  count = len([i for i in hists_flagged_per_run if i > N_bad_hists])
  count_per_run = count / len(Fdf['run_number'])
  return count_per_run

def main():

  sse_df = pd.read_csv("merged_df.csv")
  betab_df = pd.read_csv("betab_df.csv")
  pca_df = pd.read_csv("HLTPhysics_PCA_180724_myfinalassessment.csv")
  sse_df = sse_df.loc[:,~sse_df.columns.duplicated()].copy()
  betab_df = betab_df.loc[:,~betab_df.columns.duplicated()].copy()
  pca_df = pca_df.loc[:,~pca_df.columns.duplicated()].copy()

  hist_cols = [col for col in sse_df.columns if 'Run summary' in col]
  bbhc = [col for col in betab_df.columns if 'Run summary' in col]
  pcahc = [col for col in pca_df.columns if 'Run summary' in col]
  hist_dict = {each_hist: "max" for each_hist in hist_cols}
  bbhd = {each_hist: "max" for each_hist in bbhc}
  pcahd = {each_hist: "max" for each_hist in pcahc}

  sse_df = sse_df.groupby(['run_number','label'])[hist_cols].agg(hist_dict).reset_index()
  betab_df = betab_df.groupby(['run_number','label'])[bbhc].agg(bbhd).reset_index()
  pca_df = pca_df.groupby(['run_number','label'])[pcahc].agg(pcahd).reset_index()

  sse_df = sse_df.sort_values(['label']).reset_index()
  sse_df = sse_df[['run_number','label'] + [col for col in sse_df.columns if (col != 'run_number')&(col != 'label')]]
  betab_df = betab_df.sort_values(['label']).reset_index()
  betab_df = betab_df[['run_number','label'] + [col for col in betab_df.columns if (col != 'run_number')&(col != 'label')]]
  pca_df = pca_df.sort_values(['label']).reset_index()
  pca_df = pca_df[['run_number','label'] + [col for col in pca_df.columns if (col != 'run_number')&(col != 'label')]]

  sse_df_good = sse_df.loc[(sse_df['label'] == 0) | (sse_df['label'] == -1)].reset_index()
  sse_df_bad = sse_df.loc[sse_df['label'] == 1].reset_index()
  sse_df_good = sse_df_good[['run_number'] + hist_cols]
  sse_df_bad = sse_df_bad[['run_number'] + hist_cols]

  betab_df_good = betab_df.loc[(betab_df['label'] == 0) | (betab_df['label'] == -1)].reset_index()
  betab_df_bad = betab_df.loc[betab_df['label'] == 1].reset_index()
  betab_df_good = betab_df_good[['run_number'] + bbhc]
  betab_df_bad = betab_df_bad[['run_number'] + bbhc]

  pca_df_good = pca_df.loc[(pca_df['label'] == 0) | (pca_df['label'] == -1)].reset_index()
  pca_df_bad = pca_df.loc[pca_df['label'] == 1].reset_index()
  pca_df_good = pca_df_good[['run_number'] + pcahc]
  pca_df_bad = pca_df_bad[['run_number'] + pcahc]

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

  betabcah = []
  for histogram in bbhc:
    sse_ordered = sorted(betab_df_good[histogram], reverse=True)
    cutoff_0 = sse_ordered[0] + 0.5*(sse_ordered[0] - sse_ordered[1])
    cutoff_thresholds = []
    cutoff_thresholds.append(cutoff_0)
    for ii in range(len(sse_ordered)-1):
      cutoff_ii = 0.5*(sse_ordered[ii]+sse_ordered[ii+1])
      cutoff_thresholds.append(cutoff_ii)
    betabcah.append(cutoff_thresholds)

  betabcah = np.array(betabcah)

  pcacah = []
  for histogram in pcahc:
    sse_ordered = sorted(pca_df_good[histogram], reverse=True)
    cutoff_0 = sse_ordered[0] + 0.5*(sse_ordered[0] - sse_ordered[1])
    cutoff_thresholds = []
    cutoff_thresholds.append(cutoff_0)
    for ii in range(len(sse_ordered)-1):
      cutoff_ii = 0.5*(sse_ordered[ii]+sse_ordered[ii+1])
      cutoff_thresholds.append(cutoff_ii)
    pcacah.append(cutoff_thresholds)

  pcacah = np.array(pcacah)

  N_bad_hists = [1,3,5]
  tFRF_ROC_good_X = []
  tFRF_ROC_bad_Y = []
  betabfrgx = []
  betabfrby = []
  pcafrgx = []
  pcafrby = []

  for nbh_ii in N_bad_hists:
    tFRF_ROC_good_X_init = [0.0]
    tFRF_ROC_bad_Y_init = [0.0]
    #print(len(cutoffs_across_hists[0,:]))
    for cutoff_index in range(len(cutoffs_across_hists[0,:])):
      t_cutoff_index_g_FRF_rc = count_fraction_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index], nbh_ii)
      t_cutoff_index_b_FRF_rc = count_fraction_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index], nbh_ii)
      tFRF_ROC_good_X_init.append(t_cutoff_index_g_FRF_rc)
      tFRF_ROC_bad_Y_init.append(t_cutoff_index_b_FRF_rc)

    tFRF_ROC_good_X_init = sorted(tFRF_ROC_good_X_init)
    tFRF_ROC_bad_Y_init = sorted(tFRF_ROC_bad_Y_init)

    betabfrgx_init = [0.0]
    betabfrby_init = [0.0]
    for cutoff_index in range(len(betabcah[0,:])):
      t_cutoff_index_g_FRF_rc = count_fraction_runs_above(betab_df_good, betabcah[:,cutoff_index], nbh_ii)
      t_cutoff_index_b_FRF_rc = count_fraction_runs_above(betab_df_bad, betabcah[:,cutoff_index], nbh_ii)
      betabfrgx_init.append(t_cutoff_index_g_FRF_rc)
      betabfrby_init.append(t_cutoff_index_b_FRF_rc)

    betabfrgx_init = sorted(betabfrgx_init)
    betabfrby_init = sorted(betabfrby_init)

    pcafrgx_init = [0.0]
    pcafrby_init = [0.0]
    for cutoff_index in range(len(pcacah[0,:])):
      t_cutoff_index_g_FRF_rc = count_fraction_runs_above(pca_df_good, pcacah[:,cutoff_index], nbh_ii)
      t_cutoff_index_b_FRF_rc = count_fraction_runs_above(pca_df_bad, pcacah[:,cutoff_index], nbh_ii)
      pcafrgx_init.append(t_cutoff_index_g_FRF_rc)
      pcafrby_init.append(t_cutoff_index_b_FRF_rc)

    pcafrgx_init = sorted(pcafrgx_init)
    pcafrby_init = sorted(pcafrby_init)

    tFRF_ROC_good_X.append(tFRF_ROC_good_X_init)
    tFRF_ROC_bad_Y.append(tFRF_ROC_bad_Y_init)

    betabfrgx.append(betabfrgx_init)
    betabfrby.append(betabfrby_init)

    pcafrgx.append(pcafrgx_init)
    pcafrby.append(pcafrby_init)
  
  print("RF Info\n\n")
  print("Combined PCA + beta bin")
  print(tFRF_ROC_good_X)
  #print(len(tFRF_ROC_good_X[0]))
  print("\n")
  print(tFRF_ROC_bad_Y)
  print("\n\nCombined beta bin")
  print(betabfrgx)
  print("\n")
  #print(len(betabfrgx[0]))
  print(betabfrby)
  print("\n\nPCA")
  print(pcafrgx)
  print("\n")
  #print(len(pcafrgx[0]))
  print(pcafrby)
  print("\n\n\n")

  tMHF_ROC_good_X = [0.0]
  tMHF_ROC_bad_Y = [0.0]
  for cutoff_index in range(len(cutoffs_across_hists[0,:])):
    t_cutoff_index_g_MHF_rc = count_mean_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index])
    t_cutoff_index_b_MHF_rc = count_mean_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index])
    tMHF_ROC_good_X.append(t_cutoff_index_g_MHF_rc)
    tMHF_ROC_bad_Y.append(t_cutoff_index_b_MHF_rc)

  tMHF_ROC_good_X = sorted(tMHF_ROC_good_X)
  tMHF_ROC_bad_Y = sorted(tMHF_ROC_bad_Y)

  tMHF_ROC_good_X_betab = [0.0]
  tMHF_ROC_bad_Y_betab = [0.0]
  for cutoff_index in range(len(betabcah[0,:])):
    t_cutoff_index_g_MHF_rc = count_mean_runs_above(sse_df_good, betabcah[:,cutoff_index])
    t_cutoff_index_b_MHF_rc = count_mean_runs_above(sse_df_bad, betabcah[:,cutoff_index])
    tMHF_ROC_good_X_betab.append(t_cutoff_index_g_MHF_rc)
    tMHF_ROC_bad_Y_betab.append(t_cutoff_index_b_MHF_rc)

  tMHF_ROC_good_X_betab = sorted(tMHF_ROC_good_X_betab)
  tMHF_ROC_bad_Y_betab = sorted(tMHF_ROC_bad_Y_betab)

  tMHF_ROC_good_X_pca = [0.0]
  tMHF_ROC_bad_Y_pca = [0.0]
  for cutoff_index in range(len(pcacah[0,:])):
    t_cutoff_index_g_MHF_rc = count_mean_runs_above(sse_df_good, pcacah[:,cutoff_index])
    t_cutoff_index_b_MHF_rc = count_mean_runs_above(sse_df_bad, pcacah[:,cutoff_index])
    tMHF_ROC_good_X_pca.append(t_cutoff_index_g_MHF_rc)
    tMHF_ROC_bad_Y_pca.append(t_cutoff_index_b_MHF_rc)

  tMHF_ROC_good_X_pca = sorted(tMHF_ROC_good_X_pca)
  tMHF_ROC_bad_Y_pca = sorted(tMHF_ROC_bad_Y_pca)

  print("Mean Info\n\n")
  print("Combined PCA + beta bin")
  print(tMHF_ROC_good_X)
  #print(len(tFRF_ROC_good_X[0]))
  print("\n")
  print(tMHF_ROC_bad_Y)
  print("\n\nCombined beta bin")
  print(tMHF_ROC_good_X_betab)
  #print(len(betabfrgx[0]))
  print("\n")
  print(tMHF_ROC_bad_Y_betab)
  print("\n\nPCA")
  print(tMHF_ROC_good_X_pca)
  print("\n")
  #print(len(pcafrgx[0]))
  print(tMHF_ROC_bad_Y_pca)

  fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(8,6))

  axs.set_xlabel('Fraction of good runs with at least 3 histogram flags')
  axs.set_ylabel('Fraction of bad runs with at least 3 histogram flags')
  #print(N_bad_hists[jj])
  #print(tFRF_ROC_good_X[jj])
  #print(tFRF_ROC_bad_Y[jj])

  #print("hf_roc_good hf_roc_bad rf_n1_roc_good rf_n1_roc_bad rf_n3_roc_good rf_n3_roc_bad rf_n5_roc_good rf_n5_roc_bad")
  #for i in range(len(tMHF_ROC_good_X)):
  #  print(tMHF_ROC_good_X[i],tMHF_ROC_bad_Y[i],tFRF_ROC_good_X[2][i],tFRF_ROC_bad_Y[2][i],tFRF_ROC_good_X[1][i],tFRF_ROC_bad_Y[1][i],tFRF_ROC_good_X[0][i],tFRF_ROC_bad_Y[0][i])

  axs.plot(tFRF_ROC_good_X[0],tFRF_ROC_bad_Y[0], '-bo', mfc='yellow', mec='k', markersize=8, linewidth=1, label='Combined PCA + beta binomial')
  axs.plot(betabfrgx[0],betabfrby[0], '-rD', mfc='purple', mec='k', markersize=8, linewidth=1, label='Combined beta binomial')
  axs.plot(pcafrgx[0],pcafrby[0], '-g^', mfc='orange', mec='k', markersize=8, linewidth=1, label='PCA')
  axs.axis(xmin=0,xmax=0.4,ymin=0,ymax=0.8)
  axs.axline((0, 0), slope=1, linestyle='--',linewidth=0.8,color='gray')
  axs.annotate("Combined RF ROC", xy=(0.05, 0.95), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=12, weight='bold')
  axs.legend(loc='lower right')

  plt.savefig("./merged_ROC_HF_RF.pdf",bbox_inches='tight')
  print("SAVED: ./merged_ROC_HF_RF.pdf")

if __name__ == "__main__":
  main()
