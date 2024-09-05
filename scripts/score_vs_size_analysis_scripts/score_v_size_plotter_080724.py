import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ind = 0

hists = ["L1T//Run summary/L1TStage2CaloLayer1/ecalOccupancy","L1T//Run summary/L1TStage2CaloLayer1/hcalOccupancy","L1T//Run summary/L1TStage2uGMT/ugmtMuonPt","L1T//Run summary/L1TStage2uGMT/ugmtMuonEta","L1T//Run summary/L1TStage2uGMT/ugmtMuonPhi","L1T//Run summary/L1TStage2uGMT/ugmtMuonPhivsEta","L1T//Run summary/L1TStage2CaloLayer2/Central-Jets/CenJetsOcc","L1T//Run summary/L1TStage2CaloLayer2/Central-Jets/CenJetsRank","L1T//Run summary/L1TStage2CaloLayer2/Forward-Jets/ForJetsOcc","L1T//Run summary/L1TStage2CaloLayer2/Forward-Jets/ForJetsRank","L1T//Run summary/L1TStage2CaloLayer2/Isolated-EG/IsoEGsOcc","L1T//Run summary/L1TStage2CaloLayer2/Isolated-EG/IsoEGsRank","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-EG/NonIsoEGsOcc","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-EG/NonIsoEGsRank","L1T//Run summary/L1TStage2CaloLayer2/Isolated-Tau/IsoTausOcc","L1T//Run summary/L1TStage2CaloLayer2/Isolated-Tau/IsoTausRank","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-Tau/TausOcc","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-Tau/TausRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/ETTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/ETTEMRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/HTTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METHFRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METHFPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTHFRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTHFPhi","L1T//Run summary/L1TStage2BMTF/bmtf_hwPt","L1T//Run summary/L1TStage2uGMT/BMTFInput/ugmtBMTFhwEta","L1T//Run summary/L1TStage2uGMT/BMTFInput/ugmtBMTFglbhwPhi","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFhwPt","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFhwEta","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFglbhwPhiPos","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFglbhwPhiNeg","L1T//Run summary/L1TStage2uGMT/ugmtMuonPhiEmtf","L1T//Run summary/L1TStage2EMTF/cscLCTOccupancy","L1T//Run summary/L1TObjects/L1TMuon/timing/First_bunch/muons_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TMuon/timing/Last_bunch/muons_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TTau/timing/First_bunch/tau_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TTau/timing/Last_bunch/tau_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TJet/timing/First_bunch/jet_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TJet/timing/Last_bunch/jet_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_10p0_gev/egamma_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_10p0_gev/egamma_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_20p0_gev/egamma_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_20p0_gev/egamma_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_30p0_gev/egamma_eta_phi_bx_firstbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_30p0_gev/egamma_eta_phi_bx_lastbunch_0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_10p0_gev/egamma_iso_bx_ieta_firstbunch_ptmin10p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_10p0_gev/egamma_noniso_bx_ieta_firstbunch_ptmin10p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_10p0_gev/egamma_iso_bx_ieta_lastbunch_ptmin10p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_10p0_gev/egamma_noniso_bx_ieta_lastbunch_ptmin10p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_20p0_gev/egamma_iso_bx_ieta_firstbunch_ptmin20p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_20p0_gev/egamma_noniso_bx_ieta_firstbunch_ptmin20p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_20p0_gev/egamma_iso_bx_ieta_lastbunch_ptmin20p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_20p0_gev/egamma_noniso_bx_ieta_lastbunch_ptmin20p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_30p0_gev/egamma_iso_bx_ieta_firstbunch_ptmin30p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/First_bunch/ptmin_30p0_gev/egamma_noniso_bx_ieta_firstbunch_ptmin30p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_30p0_gev/egamma_iso_bx_ieta_lastbunch_ptmin30p0","L1T//Run summary/L1TObjects/L1TEGamma/timing/Last_bunch/ptmin_30p0_gev/egamma_noniso_bx_ieta_lastbunch_ptmin30p0"]
od_hists = ["L1T//Run summary/L1TStage2uGMT/ugmtMuonPt","L1T//Run summary/L1TStage2uGMT/ugmtMuonEta","L1T//Run summary/L1TStage2uGMT/ugmtMuonPhi","L1T//Run summary/L1TStage2CaloLayer2/Central-Jets/CenJetsRank","L1T//Run summary/L1TStage2CaloLayer2/Forward-Jets/ForJetsRank","L1T//Run summary/L1TStage2CaloLayer2/Isolated-EG/IsoEGsRank","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-EG/NonIsoEGsRank","L1T//Run summary/L1TStage2CaloLayer2/Isolated-Tau/IsoTausRank","L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-Tau/TausRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/ETTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/ETTEMRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/HTTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METHFRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/METHFPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTPhi","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTHFRank","L1T//Run summary/L1TStage2CaloLayer2/Energy-Sums/MHTHFPhi","L1T//Run summary/L1TStage2BMTF/bmtf_hwPt","L1T//Run summary/L1TStage2uGMT/BMTFInput/ugmtBMTFhwEta","L1T//Run summary/L1TStage2uGMT/BMTFInput/ugmtBMTFglbhwPhi","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFhwPt","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFhwEta","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFglbhwPhiPos","L1T//Run summary/L1TStage2uGMT/OMTFInput/ugmtOMTFglbhwPhiNeg","L1T//Run summary/L1TStage2uGMT/ugmtMuonPhiEmtf"]
set1 = set(hists)
set2 = set(od_hists)
td_hists = list(set1 - set2)
print(od_hists)
print(td_hists)
score_cols = ["scoreXnBins_","score_","chi2_tol0","chi2_tol1","maxpull_tol0","maxpull_tol1"]
nicer_looking_names_scores = ["SSE x nBins","SSE","Chi2 (tol 0.0)","Chi2 (tol 0.1)","Max pull (tol 0.0)","Max pull (tol 0.1)"]
size_cols = ["integral","dcs_cms_Nls","dcs_rec_lumi"]
indep_cols_to_keep = ["dcs_cms_Nls","dcs_rec_lumi","good_bad","test_train"]
nicer_looking_names_sizes = ["No. entries","DCS CMS no. LS","DCS recorded lumi"]

od_cols_final = od_hists + indep_cols_to_keep
td_cols_final = td_hists + indep_cols_to_keep

filenames = ["rebNO.csv","reb0p1.csv","reb0p33.csv","reb1p0.csv"]
df = pd.read_csv("./combined_data_from_GS/"+filenames[ind])

od_columns = [col for col in df.columns if any(substring in col for substring in od_cols_final)]
td_columns = [col for col in df.columns if any(substring in col for substring in td_cols_final)]

od_df = df[od_columns]
td_df = df[td_columns]

od_df_plotting_good = od_df.loc[(od_df['good_bad'] == "Good")]
od_df_plotting_bad = od_df.loc[(od_df['good_bad'] == "Bad")]
td_df_plotting_good = td_df.loc[(td_df['good_bad'] == "Good")]
td_df_plotting_bad = td_df.loc[(td_df['good_bad'] == "Bad")]

score_cols_in_oddf = [col for col in od_df_plotting_good.columns if any(substring in col for substring in score_cols)]
score_cols_in_tddf = [col for col in td_df_plotting_good.columns if any(substring in col for substring in score_cols)]
y_vals_oddf_good = od_df_plotting_good[score_cols_in_oddf]
y_vals_oddf_bad = od_df_plotting_bad[score_cols_in_oddf]
y_vals_tddf_good = td_df_plotting_good[score_cols_in_tddf]
y_vals_tddf_bad = td_df_plotting_bad[score_cols_in_tddf]

size_cols_in_oddf = [col for col in od_df_plotting_good.columns if any(substring in col for substring in size_cols)]
size_cols_in_tddf = [col for col in td_df_plotting_good.columns if any(substring in col for substring in size_cols)]
x_vals_oddf_good = od_df_plotting_good[size_cols_in_oddf]
x_vals_oddf_bad = od_df_plotting_bad[size_cols_in_oddf]
x_vals_tddf_good = td_df_plotting_good[size_cols_in_tddf]
x_vals_tddf_bad = td_df_plotting_bad[size_cols_in_tddf]

# good 1d
figgood1d, axsgood1d = plt.subplots(nrows=6, ncols=3, figsize=(15, 30))

for i in range(len(score_cols)):
    for j in range(len(size_cols)):
        title = score_cols[i] + " v " + size_cols[j]
        #axsgood1d[i, j].set_title(title)
        size_vals_for_plot = [col for col in x_vals_oddf_good.columns if size_cols[j] in col]
        x_vals_all_cols = x_vals_oddf_good[size_vals_for_plot]
        score_vals_for_plot = [col for col in y_vals_oddf_good.columns if score_cols[i] in col]
        y_vals_all_cols = y_vals_oddf_good[score_vals_for_plot]
        if len(size_vals_for_plot) != 1:
            for hist_col_iter in range(y_vals_all_cols.shape[1]):
                axsgood1d[i, j].scatter(x_vals_all_cols.iloc[:, hist_col_iter], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
        else:
            for hist_col_iter in range(y_vals_all_cols.shape[1]):
                axsgood1d[i, j].scatter(x_vals_all_cols.iloc[:, 0], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
        axsgood1d[i, j].set_xlabel(nicer_looking_names_sizes[j])
        axsgood1d[i, j].set_ylabel(nicer_looking_names_scores[i])
        axsgood1d[i, j].grid(True)
        axsgood1d[i, j].set_xscale('log')
        axsgood1d[i, j].set_yscale('log')
        
plt.tight_layout()
figgood1d.savefig("figgood1d"+filenames[ind][:-4]+".pdf", format='pdf')
print("SAVED: figgood1d"+filenames[ind][:-4]+".pdf")

# bad 1d
figbad1d, axsbad1d = plt.subplots(nrows=6, ncols=3, figsize=(15, 30))

for i in range(len(score_cols)):
    for j in range(len(size_cols)):
        title = score_cols[i] + " v " + size_cols[j]
        size_vals_for_plot = [col for col in x_vals_oddf_bad.columns if size_cols[j] in col]
        x_vals_all_cols = x_vals_oddf_bad[size_vals_for_plot]
        score_vals_for_plot = [col for col in y_vals_oddf_bad.columns if score_cols[i] in col]
        y_vals_all_cols = y_vals_oddf_bad[score_vals_for_plot]
        if len(size_vals_for_plot) != 1:
            for hist_col_iter in range(y_vals_all_cols.shape[1]):
                axsbad1d[i, j].scatter(x_vals_all_cols.iloc[:, hist_col_iter], y_vals_all_cols.iloc[:, hist_col_iter], c="r", s=8)
        else:
            for hist_col_iter in range(y_vals_all_cols.shape[1]):
                axsbad1d[i, j].scatter(x_vals_all_cols.iloc[:, 0], y_vals_all_cols.iloc[:, hist_col_iter], c="r", s=8)
        axsbad1d[i, j].set_xlabel(nicer_looking_names_sizes[j])
        axsbad1d[i, j].set_ylabel(nicer_looking_names_scores[i])
        axsbad1d[i, j].grid(True)
        axsbad1d[i, j].set_xscale('log')
        axsbad1d[i, j].set_yscale('log')


plt.tight_layout()
figbad1d.savefig("figbad1d"+filenames[ind][:-4]+".pdf", format='pdf')
print("SAVED: figbad1d"+filenames[ind][:-4]+".pdf")

# good 2d
figgood2d, axsgood2d = plt.subplots(nrows=6, ncols=3, figsize=(15, 30))

for i in range(len(score_cols)):
    for j in range(len(size_cols)):
        title = score_cols[i] + " v " + size_cols[j]
        size_vals_for_plot = [col for col in x_vals_tddf_good.columns if size_cols[j] in col]
        x_vals_all_cols = x_vals_tddf_good[size_vals_for_plot]
        score_vals_for_plot = [col for col in y_vals_tddf_good.columns if score_cols[i] in col]
        y_vals_all_cols = y_vals_tddf_good[score_vals_for_plot]
        if len(size_vals_for_plot) != 1:
            for hist_col_iter in range(y_vals_all_cols.shape[1]):
                axsgood2d[i, j].scatter(x_vals_all_cols.iloc[:, hist_col_iter], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
        else:
            for hist_col_iter in range(y_vals_all_cols.shape[1]):
                axsgood2d[i, j].scatter(x_vals_all_cols.iloc[:, 0], y_vals_all_cols.iloc[:, hist_col_iter], c="g", s=8)
        axsgood2d[i, j].set_xlabel(nicer_looking_names_sizes[j])
        axsgood2d[i, j].set_ylabel(nicer_looking_names_scores[i])
        axsgood2d[i, j].grid(True)
        axsgood2d[i, j].set_xscale('log')
        axsgood2d[i, j].set_yscale('log')


plt.tight_layout()
figgood2d.savefig("figgood2d"+filenames[ind][:-4]+".pdf", format='pdf')
print("SAVED: figgood2d"+filenames[ind][:-4]+".pdf")

# bad 2d
figbad2d, axsbad2d = plt.subplots(nrows=6, ncols=3, figsize=(15, 30))

for i in range(len(score_cols)):
    for j in range(len(size_cols)):
        title = score_cols[i] + " v " + size_cols[j]
        size_vals_for_plot = [col for col in x_vals_tddf_bad.columns if size_cols[j] in col]
        x_vals_all_cols = x_vals_tddf_bad[size_vals_for_plot]
        score_vals_for_plot = [col for col in y_vals_tddf_bad.columns if score_cols[i] in col]
        y_vals_all_cols = y_vals_tddf_bad[score_vals_for_plot]
        if len(size_vals_for_plot) != 1:
            for hist_col_iter in range(y_vals_all_cols.shape[1]):
                axsbad2d[i, j].scatter(x_vals_all_cols.iloc[:, hist_col_iter], y_vals_all_cols.iloc[:, hist_col_iter], c="r", s=8)
        else:
            for hist_col_iter in range(y_vals_all_cols.shape[1]):
                axsbad2d[i, j].scatter(x_vals_all_cols.iloc[:, 0], y_vals_all_cols.iloc[:, hist_col_iter], c="r", s=8)
        axsbad2d[i, j].set_xlabel(nicer_looking_names_sizes[j])
        axsbad2d[i, j].set_ylabel(nicer_looking_names_scores[i])
        axsbad2d[i, j].grid(True)
        axsbad2d[i, j].set_xscale('log')
        axsbad2d[i, j].set_yscale('log')


plt.tight_layout()
figbad2d.savefig("figbad2d"+filenames[ind][:-4]+".pdf", format='pdf')
print("SAVED: figbad2d"+filenames[ind][:-4]+".pdf")









