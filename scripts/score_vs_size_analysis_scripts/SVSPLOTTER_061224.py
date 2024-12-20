import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
ind = 1

hists = ["L1T//Run summary/L1TStage2CaloLayer1/ecalOccupancy","L1T//Run summary/L1TStage2CaloLayer1/hcalOccupancy","L1T//Run summary/L1TStage2uGMT/ugmtMuonPt","L1T//Run summary/L1TStage2uGMT/ugmtMuonEta","L1T//Run summary/L1TStage2uGMT/ugmtMuonPhi","L1T//Run summary/L1TStage2uGMT/ugmtMuonPhivsEta","L1T//Run summary/L1TStage2CaloLayer2/Central-Jets/CenJetsOcc","L1T//Run summary/L1TStage2CaloLayer2/Central-Jets/CenJetsRank","L1T//Run summary/L1TStage2CaloLayer2/Forward-Jets/ForJetsOcc","L1T//Run summary/L1TStage2CaloLayer2/Forward-Jets/ForJetsRank","L1T//Run summary/L1TStage2CaloLayer2/Isolated-EG/IsoEGsOcc","L1T//Run summary/L1TStage2CaloLayer2/Isolated-EG/IsoEGsRank","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-EG/NonIsoEGsOcc","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-EG/NonIsoEGsRank","L1T//Run summary/L1TStage2CaloLayer2/Isolated-Tau/IsoTausOcc","L1T//Run summary/L1TStage2CaloLayer2/Isolated-Tau/IsoTausRank","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-Tau/TausOcc","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-Tau/TausRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/ETTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/ETTEMRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/HTTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METHFRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METHFPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTHFRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTHFPhi","L1T//Run summary/L1TStage2BMTF/bmtf_hwPt","L1T//Run summary/L1TStage2uGMT/BMTFInput/ugmtBMTFhwEta","L1T//Run summary/L1TStage2uGMT/BMTFInput/ugmtBMTFglbhwPhi","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFhwPt","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFhwEta","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFglbhwPhiPos","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFglbhwPhiNeg","L1T//Run summary/L1TStage2uGMT/ugmtMuonPhiEmtf","L1T//Run summary/L1TStage2EMTF/cscLCTOccupancy","L1T//Run summary/L1TObjects/L1TMuon/timing/First_bunch/muons_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TMuon/timing/Last_bunch/muons_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TTau/timing/First_bunch/tau_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TTau/timing/Last_bunch/tau_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TJet/timing/First_bunch/jet_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TJet/timing/Last_bunch/jet_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_10p0_gev/egamma_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_10p0_gev/egamma_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_20p0_gev/egamma_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_20p0_gev/egamma_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_30p0_gev/egamma_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_30p0_gev/egamma_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_10p0_gev/egamma_iso_bx_ieta_firstbunch_ptmin10p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_10p0_gev/egamma_noniso_bx_ieta_firstbunch_ptmin10p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_10p0_gev/egamma_iso_bx_ieta_lastbunch_ptmin10p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_10p0_gev/egamma_noniso_bx_ieta_lastbunch_ptmin10p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_20p0_gev/egamma_iso_bx_ieta_firstbunch_ptmin20p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_20p0_gev/egamma_noniso_bx_ieta_firstbunch_ptmin20p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_20p0_gev/egamma_iso_bx_ieta_lastbunch_ptmin20p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_20p0_gev/egamma_noniso_bx_ieta_lastbunch_ptmin20p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_30p0_gev/egamma_iso_bx_ieta_firstbunch_ptmin30p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_30p0_gev/egamma_noniso_bx_ieta_firstbunch_ptmin30p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_30p0_gev/egamma_iso_bx_ieta_lastbunch_ptmin30p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_30p0_gev/egamma_noniso_bx_ieta_lastbunch_ptmin30p0"]
od_hists = ["L1T//Run summary/L1TStage2uGMT/ugmtMuonPt","L1T//Run summary/L1TStage2uGMT/ugmtMuonEta","L1T//Run summary/L1TStage2uGMT/ugmtMuonPhi","L1T//Run summary/L1TStage2CaloLayer2/Central-Jets/CenJetsRank","L1T//Run summary/L1TStage2CaloLayer2/Forward-Jets/ForJetsRank","L1T//Run summary/L1TStage2CaloLayer2/Isolated-EG/IsoEGsRank","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-EG/NonIsoEGsRank","L1T//Run summary/L1TStage2CaloLayer2/Isolated-Tau/IsoTausRank","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-Tau/TausRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/ETTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/ETTEMRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/HTTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METHFRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METHFPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTHFRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTHFPhi","L1T//Run summary/L1TStage2BMTF/bmtf_hwPt","L1T//Run summary/L1TStage2uGMT/BMTFInput/ugmtBMTFhwEta","L1T//Run summary/L1TStage2uGMT/BMTFInput/ugmtBMTFglbhwPhi","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFhwPt","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFhwEta","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFglbhwPhiPos","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFglbhwPhiNeg","L1T//Run summary/L1TStage2uGMT/ugmtMuonPhiEmtf"]
set1 = set(hists)
set2 = set(od_hists)
td_hists = list(set1 - set2)
#print(hists)

'''
# Standard 1D v 2D
score_cols = ["_score_","_chi2_tol0","_chi2prime"]
nicer_looking_names_scores = ["SSE score","Chi2","Mod. Chi2"]
size_cols = ["_integral","_integral","_integral"]
indep_cols_to_keep = ["run_number","label"]
#indep_cols_to_keep = ["dcs_cms_Nls","dcs_rec_lumi","good_bad","test_train"]
nicer_looking_names_sizes = ["No. entries","No. entries","No. entries"]
'''

# Rank 1D v 2D
score_cols = ["_score","_chi2prime"]
nicer_looking_names_scores = ["Rank SSE score","Rank mod. Chi2"]
size_cols = ["_integral","_integral"]
indep_cols_to_keep = ["run_number","label"]
#indep_cols_to_keep = ["dcs_cms_Nls","dcs_rec_lumi","good_bad","test_train"]
nicer_looking_names_sizes = ["Rank no. entries","Rank no. entries"]

od_cols_final = od_hists + indep_cols_to_keep
td_cols_final = td_hists + indep_cols_to_keep

df1 = pd.read_csv("HLTPhysics_PCA_131224_sse_scores.csv")
df1drop_cols = ["_size","_scoreXnBins","year"]
df1 = df1.loc[:, ~df1.columns.str.contains('|'.join(df1drop_cols))]

df2 = pd.read_csv("HLTPhysics_PCA_131224_modified_chi2_values.csv")
df2drop_cols = ["_size","_integral","year","algo"]
df2 = df2.loc[:, ~df2.columns.str.contains('|'.join(df2drop_cols))]

df3 = pd.read_csv("HLTPhysics_PCA_131224_chi2_maxpull_values.csv")
df3drop_cols = ["_size","_integral","year","algo","_maxpull_","_chi2tol1","_original","prediction"]
df3 = df3.loc[:, ~df3.columns.str.contains('|'.join(df3drop_cols))]

mergeddf1df2 = pd.merge(df1, df2, on=['run_number', 'label'], how='inner')
df = pd.merge(mergeddf1df2, df3, on=['run_number', 'label'], how='inner')

od_columns = [col for col in df.columns if any(substring in col for substring in od_cols_final)]
td_columns = [col for col in df.columns if any(substring in col for substring in td_cols_final)]

od_df = df[od_columns]
td_df = df[td_columns]

od_df_plotting_good = od_df.loc[(od_df['label'] == 0) | (od_df['label'] == -1)]
td_df_plotting_good = td_df.loc[(td_df['label'] == 0) | (td_df['label'] == -1)]

print(len(od_df_plotting_good))
print(len(td_df_plotting_good))


'''
# Standard plots

score_cols_in_oddf = [col for col in od_df_plotting_good.columns if any(substring in col for substring in score_cols)]
score_cols_in_tddf = [col for col in td_df_plotting_good.columns if any(substring in col for substring in score_cols)]
y_vals_oddf_good = od_df_plotting_good[score_cols_in_oddf] #.rank()
y_vals_tddf_good = td_df_plotting_good[score_cols_in_tddf] #.rank()

size_cols_in_oddf = [col for col in od_df_plotting_good.columns if any(substring in col for substring in size_cols)]
size_cols_in_tddf = [col for col in td_df_plotting_good.columns if any(substring in col for substring in size_cols)]
x_vals_oddf_good = od_df_plotting_good[size_cols_in_oddf] #.rank()
x_vals_tddf_good = td_df_plotting_good[size_cols_in_tddf] #.rank()
'''

# Rank plots

score_cols_in_oddf = [col for col in od_df_plotting_good.columns if any(substring in col for substring in score_cols)]
score_cols_in_tddf = [col for col in td_df_plotting_good.columns if any(substring in col for substring in score_cols)]
y_vals_oddf_good = od_df_plotting_good[score_cols_in_oddf].rank()
y_vals_tddf_good = td_df_plotting_good[score_cols_in_tddf].rank()

size_cols_in_oddf = [col for col in od_df_plotting_good.columns if any(substring in col for substring in size_cols)]
size_cols_in_tddf = [col for col in td_df_plotting_good.columns if any(substring in col for substring in size_cols)]
x_vals_oddf_good = od_df_plotting_good[size_cols_in_oddf].rank()
x_vals_tddf_good = td_df_plotting_good[size_cols_in_tddf].rank()

'''
for score_iter in range(len(score_cols)):
  figgood1d, axsgood1d = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    
  size_vals_for_plot = [col for col in x_vals_oddf_good.columns if size_cols[score_iter] in col]
  x_vals_all_cols = x_vals_oddf_good[size_vals_for_plot]
  score_vals_for_plot = [col for col in y_vals_oddf_good.columns if score_cols[score_iter] in col]
  y_vals_all_cols = y_vals_oddf_good[score_vals_for_plot]

  if len(size_vals_for_plot) != 1:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
      axsgood1d.scatter(x_vals_all_cols.iloc[:, hist_col_iter], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
  else:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
      axsgood1d.scatter(x_vals_all_cols.iloc[:, 0], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
  axsgood1d.set_xlabel(nicer_looking_names_sizes[score_iter])
  axsgood1d.set_ylabel(nicer_looking_names_scores[score_iter])
  axsgood1d.grid(True)
  axsgood1d.set_xscale('log')
  axsgood1d.set_yscale('log')

  plt.tight_layout()
  figgood1d.savefig("score_v_size_good_1donly"+score_cols[score_iter]+size_cols[score_iter]+".pdf", format='pdf')
  print("SAVED: score_v_size_good_1donly"+score_cols[score_iter]+size_cols[score_iter]+".pdf")

  figgood2d, axsgood2d = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
  size_vals_for_plot = [col for col in x_vals_tddf_good.columns if size_cols[score_iter] in col]
  x_vals_all_cols = x_vals_tddf_good[size_vals_for_plot]
  score_vals_for_plot = [col for col in y_vals_tddf_good.columns if score_cols[score_iter] in col]
  y_vals_all_cols = y_vals_tddf_good[score_vals_for_plot]
  if len(size_vals_for_plot) != 1:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
      axsgood2d.scatter(x_vals_all_cols.iloc[:, hist_col_iter], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
  else:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
      axsgood2d.scatter(x_vals_all_cols.iloc[:, 0], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
  axsgood2d.set_xlabel(nicer_looking_names_sizes[score_iter])
  axsgood2d.set_ylabel(nicer_looking_names_scores[score_iter])
  axsgood2d.grid(True)
  axsgood2d.set_xscale('log')
  axsgood2d.set_yscale('log')

  plt.tight_layout()
  figgood2d.savefig("score_v_size_good_2donly"+score_cols[score_iter]+size_cols[score_iter]+".pdf", format='pdf')
  print("SAVED: score_v_size_good_2donly"+score_cols[score_iter]+size_cols[score_iter]+".pdf")

'''
for score_iter in range(len(score_cols)):
  figgood1d, axsgood1d = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

  size_vals_for_plot = [col for col in x_vals_oddf_good.columns if size_cols[score_iter] in col]
  x_vals_all_cols = x_vals_oddf_good[size_vals_for_plot]
  score_vals_for_plot = [col for col in y_vals_oddf_good.columns if score_cols[score_iter] in col]
  y_vals_all_cols = y_vals_oddf_good[score_vals_for_plot]

  if len(size_vals_for_plot) != 1:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
      axsgood1d.scatter(x_vals_all_cols.iloc[:, hist_col_iter], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
  else:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
      axsgood1d.scatter(x_vals_all_cols.iloc[:, 0], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
  axsgood1d.set_xlabel(nicer_looking_names_sizes[score_iter])
  axsgood1d.set_ylabel(nicer_looking_names_scores[score_iter])
  axsgood1d.grid(True)
        
  plt.tight_layout()
  figgood1d.savefig("score_v_size_good_1donly"+score_cols[score_iter]+size_cols[score_iter]+"_rank.pdf", format='pdf')
  print("SAVED: score_v_size_good_1donly"+score_cols[score_iter]+size_cols[score_iter]+"_rank.pdf")

  figgood2d, axsgood2d = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
  size_vals_for_plot = [col for col in x_vals_tddf_good.columns if size_cols[score_iter] in col]
  x_vals_all_cols = x_vals_tddf_good[size_vals_for_plot]
  score_vals_for_plot = [col for col in y_vals_tddf_good.columns if score_cols[score_iter] in col]
  y_vals_all_cols = y_vals_tddf_good[score_vals_for_plot]

  if len(size_vals_for_plot) != 1:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
      axsgood2d.scatter(x_vals_all_cols.iloc[:, hist_col_iter], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
  else:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
      axsgood2d.scatter(x_vals_all_cols.iloc[:, 0], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
  axsgood2d.set_xlabel(nicer_looking_names_sizes[score_iter])
  axsgood2d.set_ylabel(nicer_looking_names_scores[score_iter])
  axsgood2d.grid(True)

  plt.tight_layout()
  figgood2d.savefig("score_v_size_good_2donly"+score_cols[score_iter]+size_cols[score_iter]+"_rank.pdf", format='pdf')
  print("SAVED: score_v_size_good_2donly"+score_cols[score_iter]+size_cols[score_iter]+"_rank.pdf")


