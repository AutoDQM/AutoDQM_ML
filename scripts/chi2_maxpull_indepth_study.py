'''
Macro to use post-scripts/assess.py to convert the lsit of Chi2/max pull values to ROC curves for studies over a large data set
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
    less_3_count = sum(hist_sse < 3 for hist_sse in run_row)
    if (len(run_row) - less_3_count) < hist_bad_count:
      hist_bad_count = len(run_row) - less_3_count
    bad_hist_array.append(hist_bad_count)
  return bad_hist_array

# returns mean number of runs with SSE above the given threshold
def count_mean_runs_above(Fdf, Fthreshold_list, type, single, metric):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  return_this_list = []
  if (sum(hists_flagged_per_run) > 1.5 * len(Fdf['run_number'])) & (type == "good") & (single == 1):
    print("[COUNT_MEAN_RUNS_ABOVE] Tracing the "+metric+" value thresholds for each histogram at the HF data point")
    for entry in Fthreshold_list: print(str(entry) + ",")
    return_this_list = Fthreshold_list
    single = 0
  mean_hists_flagged_per_run = sum(hists_flagged_per_run) / len(Fdf['run_number'])
  return mean_hists_flagged_per_run, single, return_this_list

# returns fraction of runs with SSE above the given threshold
def count_fraction_runs_above(Fdf, Fthreshold_list, N_bad_hists, type, single, metric):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  count = len([i for i in hists_flagged_per_run if i > N_bad_hists])
  return_this_list = []
  if (N_bad_hists == 3) & (count > 0.1 * len(Fdf['run_number'])) & (type == "good") & (single == 1):
    print("[COUNT_FRACTIONS_RUNS_ABOVE] Tracing the "+metric+" value thresholds for each histogram at the RF data point")
    for entry in Fthreshold_list: print(str(entry) + ",")
    return_this_list = Fthreshold_list
    single = 0
  count_per_run = count / len(Fdf['run_number'])
  return count_per_run, single, return_this_list

def find_closest_index(arr, dpthres):
  closest_value = min(arr, key=lambda x: abs(x - dpthres))
  closest_index = arr.index(closest_value)
  return closest_index

def count_run_most_flags(df, score_thresholds):
  result_df = pd.DataFrame(columns=['run_number', 'count_exceeds'])
  for index, row in df.iterrows():
    run = row['run_number']
    thresholds_iterator = iter(score_thresholds)
    threshold = next(thresholds_iterator, None)

    count_exceeds = 0

    for histogram_column in df.columns[1:]:
        score = row[histogram_column]

        if score > threshold:
            count_exceeds += 1

        threshold = next(thresholds_iterator, None)
    result_df = result_df.append({'run_number': run, 'count_exceeds': count_exceeds}, ignore_index=True)
  result_df['run_number'] = result_df['run_number'].astype(int).astype(str) + ','
  result_df = result_df.sort_values(by='count_exceeds', ascending=False)
  return result_df

def main(args):

  val_value = "Chi2"
  #val_value = "Max pull"

  if val_value == "Chi2": score_col_coinc = "chi2_tol1"
  if val_value == "Max pull": score_col_coinc = "maxpull_tol1"

  sse_df = pd.read_csv(args.input_file)
  algorithm_name = str(sse_df['algo'].iloc[0]).upper()
  if algorithm_name == "BETAB": algorithm_name = "Beta_Binomial"

  sse_df = sse_df.loc[:,~sse_df.columns.duplicated()].copy()
  hist_cols = [col for col in sse_df.columns if score_col_coinc in col]
  hist_dict = {each_hist: "max" for each_hist in hist_cols}

  sse_df = sse_df.groupby(['run_number','label'])[hist_cols].agg(hist_dict).reset_index()
  sse_df = sse_df.sort_values(['label']).reset_index()
  sse_df = sse_df[['run_number','label'] + [col for col in sse_df.columns if (col != 'run_number')&(col != 'label')]]

  sse_df_good = sse_df.loc[sse_df['label'] == 0].reset_index()
  sse_df_bad = sse_df.loc[sse_df['label'] == 1].reset_index()
  sse_df_good = sse_df_good[['run_number'] + hist_cols]
  sse_df_bad = sse_df_bad[['run_number'] + hist_cols]

  print("[LOGGER] Calculating SSE thresholds for each histogram for algorithm :" + algorithm_name)

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
  print("[LOGGER] There are " + str(len(cutoffs_across_hists)) + " histograms, " + str(len(sse_df_good['run_number'])) + " good runs and " + str(len(sse_df_bad['run_number']))  + " bad runs")

  print("[LOGGER] Counting the number of histogram flags at RF threshold > 0.1, N = 3")

  nbh_ii = 3
  tFRF_ROC_good_X = []
  tFRF_ROC_bad_Y = []
  single = 1
  for cutoff_index in range(len(cutoffs_across_hists[0,:])):
    #if nbh_ii == 3: print("HERE:" + str(cutoff_index))
    t_cutoff_index_g_FRF_rc, single, thresholds_fr = count_fraction_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index], nbh_ii, "good", single, val_value)
    t_cutoff_index_b_FRF_rc, single, dummy_bad_thresh_fr = count_fraction_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index], nbh_ii, "bad", single, val_value)
    tFRF_ROC_good_X.append(t_cutoff_index_g_FRF_rc)
    tFRF_ROC_bad_Y.append(t_cutoff_index_b_FRF_rc)
    if single == 0: break

  if np.array_equal(np.array(thresholds_fr),[]) == True: print("[LOGGER] Error in filling thresholds for RF metric, no data point found")

  tFRF_ROC_good_X = sorted(tFRF_ROC_good_X)
  tFRF_ROC_bad_Y = sorted(tFRF_ROC_bad_Y)

  tMHF_ROC_good_X = []
  tMHF_ROC_bad_Y = []

  single = 1
  for cutoff_index in range(len(cutoffs_across_hists[0,:])):
    t_cutoff_index_g_MHF_rc, single, thresholds_mh = count_mean_runs_above(sse_df_good, cutoffs_across_hists[:,cutoff_index], "good", single, val_value)
    t_cutoff_index_b_MHF_rc, single, dummy_bad_thresh_mh = count_mean_runs_above(sse_df_bad, cutoffs_across_hists[:,cutoff_index], "bad", single, val_value)
    tMHF_ROC_good_X.append(t_cutoff_index_g_MHF_rc)
    tMHF_ROC_bad_Y.append(t_cutoff_index_b_MHF_rc)
    if single == 0: break

  if np.array_equal(np.array(thresholds_mh),[]) == True: print("[LOGGER] Error in filling thresholds for HF metric, no data point found")

  #print(tMHF_ROC_good_X)
  tMHF_ROC_good_X = sorted(tMHF_ROC_good_X)
  tMHF_ROC_bad_Y = sorted(tMHF_ROC_bad_Y)

  algo = "pca"
  metric = "fr"
  #metric = "mh"
  if metric == "fr":
    thresholds_to_study = thresholds_fr
  if metric == "mh":
    thresholds_to_study = thresholds_mh

  print("[LOGGER] Now calculating the number of histogram flags per good and bad run, with algorithm " + algo.upper() + " and metric " + metric.upper() + " options selected")

  run_flags_good = count_run_most_flags(sse_df_good,thresholds_to_study)
  run_flags_bad = count_run_most_flags(sse_df_bad,thresholds_to_study)

  print("Good")
  print(run_flags_good.to_string(index=False))

  print("Bad")
  print(run_flags_bad.to_string(index=False))

  sse_df_good_runs_only = sse_df_good.loc[(sse_df_good['run_number'] == 355913) | (sse_df_good['run_number'] == 356386) | (sse_df_good['run_number'] == 356956)]

  result_dict = {}

  for index, row in sse_df_good_runs_only.iterrows():
    run = row['run_number']
    run_data = row.drop('run_number')

    # Find histograms above the corresponding scores
    above_threshold = run_data.index[run_data > thresholds_to_study].tolist()

    # Store the result in the dictionary
    result_dict[run] = above_threshold

  result_df = pd.DataFrame(list(result_dict.items()), columns=['Run', 'Histograms'])

  # prints histograms flagged in a particular run, following selection of runs above (good runs with high numbers of flags)
  for index, row in result_df.iterrows():
    run = row['Run']
    histograms = row['Histograms']
    
    print(f"Run {run}:")
    for hist in histograms:
        print(hist[17:(-14 - len(algo))])
    print()
  

  # index of point in array closest to MH = 1.5 from good runs
  # array of Chi2/max pull values all hists and runs
  # extract Chi2/max pull values per hist, order, and calculate thresholds
  # for each threshold, find number of Chi2/max pull values above this for each hist and matching hist threshold per 265 runs, for each of good and bad run df
  # sum across runs and divide by number of runs

  result_data_mh = {'Histogram': [], 'MH': []}
  result_data_fr = {'Histogram': [], 'FR': []}
  result_data_mh_bad = {'Histogram': [], 'MH': []}
  result_data_fr_bad = {'Histogram': [], 'FR': []}

  for column, threshold in zip(sse_df.columns[1:], thresholds_mh):
    graph_name = column
    threshold_count = (sse_df[column] > threshold).sum()
    result_data_mh['Histogram'].append(graph_name)
    result_data_mh['MH'].append(threshold_count)

  result_df_mh = pd.DataFrame(result_data_mh)

  for column, threshold in zip(sse_df.columns[1:], thresholds_fr):
    graph_name = column
    threshold_count = (sse_df[column] > threshold).sum()
    result_data_fr['Histogram'].append(graph_name)
    result_data_fr['FR'].append(threshold_count)

  result_df_fr = pd.DataFrame(result_data_fr)
  result_df = pd.merge(result_df_mh, result_df_fr, on='Histogram')

  result_df.to_csv('./hist_flag_freq_all_runs_'+algo+"_"+val_value+'.csv', index=False)


  for column, threshold in zip(sse_df_bad.columns[1:], thresholds_mh):
    graph_name = column
    threshold_count = (sse_df_bad[column] > threshold).sum()
    result_data_mh_bad['Histogram'].append(graph_name)
    result_data_mh_bad['MH'].append(threshold_count)

  result_df_mh_bad = pd.DataFrame(result_data_mh_bad)
  #print(result_df_mh_bad)

  for column, threshold in zip(sse_df_bad.columns[1:], thresholds_fr):
    graph_name = column
    threshold_count = (sse_df_bad[column] > threshold).sum()
    result_data_fr_bad['Histogram'].append(graph_name)
    result_data_fr_bad['FR'].append(threshold_count)

  result_df_fr_bad = pd.DataFrame(result_data_fr_bad)
  result_df_bad = pd.merge(result_df_mh_bad, result_df_fr_bad, on='Histogram')


  print("[LOGGER] Now calculating the frequency at which histograms are flagged in bad run, as flagged by the " + algo.upper() + " algorithm")
  print("[LOGGER] Results for HF metric:")
  csv_string_df_bad_mh = result_df_bad[["MH"]].to_csv(index=False, header=False)
  print(csv_string_df_bad_mh.strip())
  print("[LOGGER] Results for RF metric:")
  csv_string_df_bad_fr = result_df_bad[["FR"]].to_csv(index=False, header=False)
  print(csv_string_df_bad_fr.strip())
  result_df_bad.to_csv('./hist_flag_freq_bad_runs_'+algo+"_"+val_value+'.csv', index=False)

  index_mh1p5 = find_closest_index(tMHF_ROC_good_X, 1.5)
  dist_of_sse_at_mh1p5 = np.array([sub_array[index_mh1p5] for sub_array in cutoffs_across_hists])
  log_scale_mh1p5 = np.log2(dist_of_sse_at_mh1p5)

  index_rf0p1 = find_closest_index(tFRF_ROC_good_X, 0.1)
  dist_of_sse_at_rf0p1 = np.array([sub_array[index_rf0p1] for sub_array in cutoffs_across_hists])
  log_scale_rf0p1 = np.log2(dist_of_sse_at_rf0p1)

  bin_width = 0.1
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.hist(log_scale_mh1p5, bins=np.arange(-5, 10, step=bin_width), alpha=0.5, edgecolor='red')
  ax1.set_title(val_value + r' distribution for MH$_{\mathrm{good}}$ = 1.5')
  ax1.set_xlabel(val_value + ' value')
  ax1.set_ylabel('Frequency')

  ax2.hist(log_scale_rf0p1, bins=np.arange(-5, 10, step=bin_width), alpha=0.5, edgecolor='green')
  ax2.set_title(val_value + r' distribution for RF$_{\mathrm{good}}$ = 0.1')
  ax2.set_xlabel(val_value + ' value')
  ax2.set_ylabel('Frequency')

  plt.tight_layout()

  plt.show()

if __name__ == "__main__":
  args = parse_arguments()
  main(args)
