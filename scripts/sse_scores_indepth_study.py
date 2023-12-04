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

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--input_file",
    help = "input file (i.e. output from fetch_data.py) to use for training the ML algorithm",
    type = str,
    required = True,
    default = None
  )
  return parser.parse_args()

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
def count_mean_runs_above(Fdf, Fthreshold_list, type):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  #if (sum(hists_flagged_per_run) > 1.5 * len(Fdf['run_number'])) & (type == "good"):
  #  print(Fthreshold_list)
  mean_hists_flagged_per_run = sum(hists_flagged_per_run) / len(Fdf['run_number'])
  #print(mean_hists_flagged_per_run)
  return mean_hists_flagged_per_run

# returns fraction of runs with SSE above the given threshold
def count_fraction_runs_above(Fdf, Fthreshold_list, N_bad_hists, type):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  count = len([i for i in hists_flagged_per_run if i > N_bad_hists])
  #if (N_bad_hists == 3) & (count > 0.1 * len(Fdf['run_number'])) & (type == "good"):
  #  print(Fthreshold_list)
  count_per_run = count / len(Fdf['run_number'])
  return count_per_run

def find_closest_index(arr, dpthres):
  closest_value = min(arr, key=lambda x: abs(x - dpthres))
  closest_index = arr.index(closest_value)
  return closest_index

def main(args):

  sse_df = pd.read_csv(args.input_file)
  algorithm_name = str(sse_df['algo'].iloc[0]).upper()
  if algorithm_name == "BETAB": algorithm_name = "Beta_Binomial"

  sse_df = sse_df.loc[:,~sse_df.columns.duplicated()].copy()
  hist_cols = [col for col in sse_df.columns if '_score_' in col]
  print(len(hist_cols))
  hist_dict = {each_hist: "max" for each_hist in hist_cols}

  sse_df = sse_df.groupby(['run_number','label'])[hist_cols].agg(hist_dict).reset_index()
  sse_df = sse_df.sort_values(['label']).reset_index()
  sse_df = sse_df[['run_number','label'] + [col for col in sse_df.columns if (col != 'run_number')&(col != 'label')]]

  sse_df_good = sse_df.loc[sse_df['label'] == 0].reset_index()
  sse_df_bad = sse_df.loc[sse_df['label'] == 1].reset_index()
  sse_df_good = sse_df_good[['run_number'] + hist_cols]
  sse_df_bad = sse_df_bad[['run_number'] + hist_cols]

  print(len(sse_df_good.columns))
  print(len(sse_df_bad.columns))

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

  N_bad_hists = [5,3,1]
  tFRF_ROC_good_X = []
  tFRF_ROC_bad_Y = []

  for nbh_ii in N_bad_hists:
    tFRF_ROC_good_X_init = []
    tFRF_ROC_bad_Y_init = []
    for cutoff_index in range(len(cutoffs_across_hists[0,:])):
      #if nbh_ii == 3: print(cutoff_index)
      t_cutoff_index_g_FRF_rc = count_fraction_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index], nbh_ii, "good")
      t_cutoff_index_b_FRF_rc = count_fraction_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index], nbh_ii, "bad")
      tFRF_ROC_good_X_init.append(t_cutoff_index_g_FRF_rc)
      tFRF_ROC_bad_Y_init.append(t_cutoff_index_b_FRF_rc)

    tFRF_ROC_good_X_init = sorted(tFRF_ROC_good_X_init)
    tFRF_ROC_bad_Y_init = sorted(tFRF_ROC_bad_Y_init)

    tFRF_ROC_good_X.append(tFRF_ROC_good_X_init)
    tFRF_ROC_bad_Y.append(tFRF_ROC_bad_Y_init)
    

  tMHF_ROC_good_X = []
  tMHF_ROC_bad_Y = []
  for cutoff_index in range(len(cutoffs_across_hists[0,:])):
    t_cutoff_index_g_MHF_rc = count_mean_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index], "good")
    t_cutoff_index_b_MHF_rc = count_mean_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index], "bad")
    tMHF_ROC_good_X.append(t_cutoff_index_g_MHF_rc)
    tMHF_ROC_bad_Y.append(t_cutoff_index_b_MHF_rc)

  tMHF_ROC_good_X = sorted(tMHF_ROC_good_X)
  tMHF_ROC_bad_Y = sorted(tMHF_ROC_bad_Y)

  # index of point in array closest to MH = 1.5 from good runs
  # array of sse scores all hists and runs
  # extract sse scores per hist, order, and calculate thresholds
  # for each threshold, find number of sse scores above this for each hist and matching hist threshold per 265 runs, for each of good and bad run df
  # sum across runs and divide by number of runs

  #thresholds_mh = [1.33880645e-06,3.70879538e-06,2.07432838e-04,2.66117342e-04,1.53229424e-04,1.50588933e-04,7.60579091e-05,2.56020621e-04,7.17893025e-04,7.51130839e-04,7.71975298e-05,1.21253016e-03,1.73886274e-04,4.68996706e-03,2.82594840e-05,4.29786637e-04,1.87090182e-04,1.22143826e-03,2.29235532e-04,2.26782300e-04,1.70092400e-04,1.90973911e-04,1.95007126e-04,2.04282311e-04,2.13468611e-04,1.69465104e-04,3.49285997e-04,1.69873415e-04,3.38607530e-04,6.96611175e-04,9.32676717e-04,1.10394778e-03,6.60170297e-04,6.76419962e-04,1.22298216e-03,1.10916584e-03,2.34616131e-04,4.75172915e-05,5.57723951e-02,5.29767061e-02,6.71458421e-03,8.19580717e-03,3.56364660e-03,4.33776876e-03,1.21058412e-02,1.42543040e-02,3.26262834e-02,3.79098123e-02,4.55855036e-02,5.45623678e-02,5.32348925e-03,3.60698042e-03,5.95885243e-03,4.54829651e-03,1.28365046e-02,1.00816028e-02,1.42640427e-02,1.16641336e-02,1.82043639e-02,1.58835050e-02,1.91096728e-02,1.83462891e-02]
  thresholds_mh = [0.00034324,0.00092252,0.01413556,0.00108673,0.00024861,0.00041836,0.00039845,0.00313177,0.00319687,0.00420914,0.00029102,0.02245633,0.00031072,0.02519195,0.00016806,0.0283848,0.00031453,0.00689271,0.00224029,0.00390414,0.02686999,0.0136855,0.00024362,0.01378011,0.00026005,0.02457584,0.00042429,0.02856238,0.00039182,0.00471469,0.0013788,0.00123662,0.00692726,0.00245624,0.0012641,0.00114116,0.00042045,0.00016676,0.07696494,0.06275035,0.00830455,0.00974406,0.00384988,0.00623181,0.01425515,0.015748,0.03580804,0.0394197,0.05351183,0.06638868,0.00622217,0.0039951,0.00663571,0.00499496,0.01498416,0.01172696,0.01673972,0.01377621,0.02252843,0.01897173,0.0216591,0.02035667]
  #thresholds_fr = [1.21923653e-06,3.39235805e-06,1.46611637e-04,1.86651626e-04,1.34228888e-04,1.34414060e-04,5.52984813e-05,2.04392598e-04,5.91147559e-04,6.57175545e-04,6.21854772e-05,1.09874233e-03,1.38764334e-04,3.04778903e-03,2.05674547e-05,3.20099752e-04,1.59766659e-04,9.72542278e-04,1.76607148e-04,1.71897154e-04,1.39978724e-04,1.52245923e-04,1.37982468e-04,1.43233418e-04,1.50150669e-04,1.28892336e-04,2.56956056e-04,1.31040269e-04,2.43951748e-04,5.71627414e-04,8.59285632e-04,9.21611290e-04,3.72258677e-04,5.72983971e-04,9.82457296e-04,9.82599023e-04,1.98513105e-04,3.86858372e-05,3.84118727e-02,4.27476439e-02,5.76547988e-03,6.60202868e-03,3.03147035e-03,3.33583893e-03,9.41221220e-03,1.21282967e-02,2.44694405e-02,3.21343906e-02,3.70272178e-02,4.99990694e-02,4.28180460e-03,3.04120583e-03,4.82706018e-03,3.27151354e-03,9.19279936e-03,7.68303605e-03,1.11989272e-02,9.55544609e-03,1.39589441e-02,1.34185967e-02,1.65757293e-02,1.60363727e-02]
  thresholds_fr = [0.00034324,0.00092252,0.01413556,0.00108673,0.00024861,0.00041836,0.00039845,0.00313177,0.00319687,0.00420914,0.00029102,0.02245633,0.00031072,0.02519195,0.00016806,0.0283848,0.00031453,0.00689271,0.00224029,0.00390414,0.02686999,0.0136855,0.00024362,0.01378011,0.00026005,0.02457584,0.00042429,0.02856238,0.00039182,0.00471469,0.0013788,0.00123662,0.00692726,0.00245624,0.0012641,0.00114116,0.00042045,0.00016676,0.07696494,0.06275035,0.00830455,0.00974406,0.00384988,0.00623181,0.01425515,0.015748,0.03580804,0.0394197,0.05351183,0.06638868,0.00622217,0.0039951,0.00663571,0.00499496,0.01498416,0.01172696,0.01673972,0.01377621,0.02252843,0.01897173,0.0216591,0.02035667]
  result_data_mh = {'Histogram': [], 'MH': []}
  result_data_fr = {'Histogram': [], 'FR': []}


  for column, threshold in zip(sse_df_bad.columns[1:], thresholds_mh):
    graph_name = column
    threshold_count = (sse_df_bad[column] > threshold).sum()
    result_data_mh['Histogram'].append(graph_name)
    result_data_mh['MH'].append(threshold_count)

  result_df_mh = pd.DataFrame(result_data_mh)
  print(result_df_mh)

  for column, threshold in zip(sse_df_bad.columns[1:], thresholds_fr):
    graph_name = column
    threshold_count = (sse_df_bad[column] > threshold).sum()
    result_data_fr['Histogram'].append(graph_name)
    result_data_fr['FR'].append(threshold_count)

  result_df_fr = pd.DataFrame(result_data_fr)
  result_df = pd.merge(result_df_mh, result_df_fr, on='Histogram')

  result_df.to_csv('./hist_flag_freq_bad_runs_ae.csv', index=False)


  index_mh1p5 = find_closest_index(tMHF_ROC_good_X, 1.5)
  dist_of_sse_at_mh1p5 = np.array([sub_array[index_mh1p5] for sub_array in cutoffs_across_hists])
  log_scale_mh1p5 = np.log10(dist_of_sse_at_mh1p5)

  index_rf0p1 = find_closest_index(tFRF_ROC_good_X[1], 0.1)
  dist_of_sse_at_rf0p1 = np.array([sub_array[index_rf0p1] for sub_array in cutoffs_across_hists])
  log_scale_rf0p1 = np.log10(dist_of_sse_at_rf0p1)

  bin_width = 0.1
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.hist(log_scale_mh1p5, bins=np.arange(-6, 0, step=bin_width), alpha=0.5, edgecolor='red')
  ax1.set_title(r'SSE distribution for MH$_{\mathrm{good}}$ = 1.5')
  ax1.set_xlabel('SSE score')
  ax1.set_ylabel('Frequency')

  ax2.hist(log_scale_rf0p1, bins=np.arange(-6, 0, step=bin_width), alpha=0.5, edgecolor='green')
  ax2.set_title(r'SSE distribution for RF$_{\mathrm{good}}$ = 0.1')
  ax2.set_xlabel('SSE score')
  ax2.set_ylabel('Frequency')

  plt.tight_layout()

  #plt.show()


if __name__ == "__main__":
  args = parse_arguments()
  main(args)
