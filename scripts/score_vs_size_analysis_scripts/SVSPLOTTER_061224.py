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

score_cols = ["chi2XNoEntriesRank"]
nicer_looking_names_scores = ["Rank mod. Chi2"]
size_cols = ["noEntriesRank"]
indep_cols_to_keep = ["dcs_cms_Nls","dcs_rec_lumi","good_bad","test_train"]
nicer_looking_names_sizes = ["Rank no. entries"]

od_cols_final = od_hists + indep_cols_to_keep
td_cols_final = td_hists + indep_cols_to_keep

filenames = ["rebNO.csv","reb0p33.csv"]
od_df = pd.read_csv("type_1d_mod_score_v_size_reb0p33.csv")
td_df = pd.read_csv("type_2d_mod_score_v_size_reb0p33.csv")

#od_columns = [col for col in df.columns if any(substring in col for substring in od_cols_final)]
#td_columns = [col for col in df.columns if any(substring in col for substring in td_cols_final)]

#od_df = df[od_columns]
#td_df = df[td_columns]

od_df_plotting_good = od_df.loc[(od_df['good_bad'] == "Good")]
td_df_plotting_good = td_df.loc[(td_df['good_bad'] == "Good")]

print(len(od_df_plotting_good))
print(len(td_df_plotting_good))

score_cols_in_oddf = [col for col in od_df_plotting_good.columns if any(substring in col for substring in score_cols)]
score_cols_in_tddf = [col for col in td_df_plotting_good.columns if any(substring in col for substring in score_cols)]
y_vals_oddf_good = od_df_plotting_good[score_cols_in_oddf].rank()
y_vals_tddf_good = td_df_plotting_good[score_cols_in_tddf].rank()
print(y_vals_tddf_good)

size_cols_in_oddf = [col for col in od_df_plotting_good.columns if any(substring in col for substring in size_cols)]
size_cols_in_tddf = [col for col in td_df_plotting_good.columns if any(substring in col for substring in size_cols)]
x_vals_oddf_good = od_df_plotting_good[size_cols_in_oddf].rank()
x_vals_tddf_good = td_df_plotting_good[size_cols_in_tddf].rank()

# good 1d
figgood1d, axsgood1d = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

#title = score_cols[0] + " v " + size_cols[0]

size_vals_for_plot = [col for col in x_vals_oddf_good.columns if size_cols[0] in col]
x_vals_all_cols = x_vals_oddf_good[size_vals_for_plot]
#x_vals_all_cols = np.argsort(np.argsort(x_vals_all_cols)) + 1
score_vals_for_plot = [col for col in y_vals_oddf_good.columns if score_cols[0] in col]
y_vals_all_cols = y_vals_oddf_good[score_vals_for_plot]
#y_vals_all_cols = np.argsort(np.argsort(y_vals_all_cols)) + 1

if len(size_vals_for_plot) != 1:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
        axsgood1d.scatter(x_vals_all_cols.iloc[:, hist_col_iter], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
else:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
        axsgood1d.scatter(x_vals_all_cols.iloc[:, 0], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
axsgood1d.set_xlabel(nicer_looking_names_sizes[0])
axsgood1d.set_ylabel(nicer_looking_names_scores[0])
axsgood1d.grid(True)
        
plt.tight_layout()
figgood1d.savefig("score_v_size_good_1donly_chi2XNoEntriesPow_modChi2_rank.pdf", format='pdf')
print("SAVED: score_v_size_good_1donly_chi2XNoEntriesPow_modChi2_rank.pdf")

# good 2d
figgood2d, axsgood2d = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

#title = score_cols[0] + " v " + size_cols[0]
size_vals_for_plot = [col for col in x_vals_tddf_good.columns if size_cols[0] in col]
x_vals_all_cols = x_vals_tddf_good[size_vals_for_plot]
#x_vals_all_cols = np.argsort(np.argsort(x_vals_all_cols)) + 1
score_vals_for_plot = [col for col in y_vals_tddf_good.columns if score_cols[0] in col]
y_vals_all_cols = y_vals_tddf_good[score_vals_for_plot]
#y_vals_all_cols = np.argsort(np.argsort(y_vals_all_cols)) + 1

if len(size_vals_for_plot) != 1:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
        axsgood2d.scatter(x_vals_all_cols.iloc[:, hist_col_iter], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
else:
    for hist_col_iter in range(y_vals_all_cols.shape[1]):
        axsgood2d.scatter(x_vals_all_cols.iloc[:, 0], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
axsgood2d.set_xlabel(nicer_looking_names_sizes[0])
axsgood2d.set_ylabel(nicer_looking_names_scores[0])
axsgood2d.grid(True)


plt.tight_layout()
figgood2d.savefig("score_v_size_good_2donly_chi2XNoEntriesPow_modChi2_rank.pdf", format='pdf')
print("SAVED: score_v_size_good_2donly_chi2XNoEntriesPow_modChi2_rank.pdf")










