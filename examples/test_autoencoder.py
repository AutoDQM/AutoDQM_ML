import pandas
import numpy as np
from scipy.stats import chisquare
#file = "CSC_EMTF_InitialList_4May2021_SingleMuon_short.pkl"
file = "output/test_SingleMuon.pkl"
df = pandas.read_pickle(file)

from autodqm_ml.utils import setup_logger
logger = setup_logger("DEBUG")

from autodqm_ml.algorithms.autoencoder import AutoEncoder
from autodqm_ml.data_formats.histogram import Histogram
from matplotlib import pyplot as plt

names = [
               #'CSC//Run summary/CSCOfflineMonitor/Occupancy/hORecHits',
               #'CSC//Run summary/CSCOfflineMonitor/Occupancy/hOSegments',
               #'CSC//Run summary/CSCOfflineMonitor/Segments/hSTimeCombinedSerial',
               #'CSC//Run summary/CSCOfflineMonitor/Segments/hSTimeVsTOF',
               #'CSC//Run summary/CSCOfflineMonitor/Segments/hSTimeVsZ',
               #'L1T//Run summary/L1TStage2EMTF/emtfTrackBX',
               #'L1T//Run summary/L1TStage2EMTF/emtfTrackEta',
               #'L1T//Run summary/L1TStage2EMTF/emtfTrackOccupancy',
               #'L1T//Run summary/L1TStage2EMTF/emtfTrackPhi',
               #'L1T//Run summary/L1TStage2EMTF/emtfTrackQualityVsMode',
               'L1T//Run summary/L1TStage2CaloLayer2/Isolated-Tau/IsoTausEta',
               'L1T//Run summary/L1TStage2CaloLayer2/Central-Jets/CenJetsEta',
               'L1T//Run summary/L1TStage2CaloLayer2/Isolated-EG/IsoEGsEta',
               'L1T//Run summary/L1TStage2CaloLayer2/Forward-Jets/ForJetsEta',
               'L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-Tau/TausEta',
               'L1T//Run summary/L1TStage2CaloLayer2/NonIsolated-EG/NonIsoEGsEta'
]

def maximum(a, b):
  if a >= b:
    return a
  else:
    return b

a = AutoEncoder(name = "test")
a.train(histograms = names, file = file, config = {}, n_epochs = 100, batch_size = 2000)
PLOT_NO = 0

n_bins = df[names[0]][1].shape[0]
sum_of_rows = df[names[0]][1].sum()
myHist = Histogram(name = names[0], data = df[names[0]][1])
myHist.normalize()
myHist.data
myReshapedHist = myHist.data.reshape(1,n_bins,1)

n_bins2 = df[names[1]][1].shape[0]
sum_of_rows2 = df[names[1]][1].sum()
myHist2 = Histogram(name = names[1], data = df[names[1]][1])
myHist2.normalize()
myHist2.data
myReshapedHist2 = myHist2.data.reshape(1,n_bins2,1)

n_bins3 = df[names[2]][1].shape[0]
sum_of_rows3 = df[names[2]][1].sum()
myHist3 = Histogram(name = names[2], data = df[names[2]][1])
myHist3.normalize()
myHist3.data
myReshapedHist3 = myHist3.data.reshape(1,n_bins3,1)

n_bins4 = df[names[3]][1].shape[0]
sum_of_rows4 = df[names[3]][1].sum()
myHist4 = Histogram(name = names[3], data = df[names[3]][1])
myHist4.normalize()
myHist4.data
myReshapedHist4 = myHist4.data.reshape(1,n_bins4,1)

n_bins5 = df[names[4]][1].shape[0]
sum_of_rows5 = df[names[4]][1].sum()
myHist5 = Histogram(name = names[4], data = df[names[4]][1])
myHist5.normalize()
myHist5.data
myReshapedHist5 = myHist5.data.reshape(1,n_bins5,1)

n_bins6 = df[names[5]][1].shape[0]
sum_of_rows6 = df[names[5]][1].sum()
myHist6 = Histogram(name = names[5], data = df[names[5]][1])
myHist6.normalize()
myHist6.data
myReshapedHist6 = myHist6.data.reshape(1,n_bins6,1)

n_bins_array = [n_bins, n_bins2, n_bins3, n_bins4, n_bins5, n_bins6]	

spacing = 56 / n_bins_array[PLOT_NO]
eta_axis = np.linspace(-28,28,num=n_bins_array[PLOT_NO])
f = plt.figure(figsize=(18,6))
ax1 = f.add_subplot(131)
ax2 = f.add_subplot(132, sharey = ax1)
ax3 = f.add_subplot(133)

sum_of_rows_array = [sum_of_rows, sum_of_rows2, sum_of_rows3, sum_of_rows4, sum_of_rows5, sum_of_rows6]

ax1.bar(eta_axis, df[names[PLOT_NO]][1],width=1)
ax1.set_title("Input")
ax1.set_xlabel(r"$\eta$")
ax1.set_ylabel("Counts")

myReshapedHists = {
    'input_L1T__Runsummary_L1TStage2CaloLayer2_Central-Jets_CenJetsEta' : myReshapedHist,
    'input_L1T__Runsummary_L1TStage2CaloLayer2_Isolated-EG_IsoEGsEta' : myReshapedHist2,
    'input_L1T__Runsummary_L1TStage2CaloLayer2_Isolated-Tau_IsoTausEta' : myReshapedHist3,
    'input_L1T__Runsummary_L1TStage2CaloLayer2_Forward-Jets_ForJetsEta' : myReshapedHist4,
    'input_L1T__Runsummary_L1TStage2CaloLayer2_NonIsolated-Tau_TausEta' : myReshapedHist5,
    'input_L1T__Runsummary_L1TStage2CaloLayer2_NonIsolated-EG_NonIsoEGsEta' : myReshapedHist6
}

reco = a.model.predict(myReshapedHists)

myReshapedReco = reco[PLOT_NO].reshape(n_bins_array[PLOT_NO],)

ax2.bar(eta_axis, myReshapedReco*sum_of_rows_array[PLOT_NO],width=1,color='orange')
ax2.set_title("Recon")
ax2.set_xlabel(r"$\eta$")

z,cov = np.polyfit(x=df[names[PLOT_NO]][1], y=myReshapedReco*sum_of_rows_array[PLOT_NO], deg=1, cov=True)
print(cov)
restored = myReshapedReco*sum_of_rows_array[PLOT_NO]
ax3.scatter(df[names[PLOT_NO]][1], restored, marker='x', color='green')
max_y = (myReshapedReco*sum_of_rows_array[PLOT_NO]).max()
max_x = df[names[PLOT_NO]][1].max()
largest_max = maximum(max_x,max_y)
xp = np.linspace(0,largest_max,n_bins_array[PLOT_NO])
yp = z[0]*xp + z[1]
#ax3.plot(xp, yp,'--')
ax3.plot(xp, xp,'--')
ax3.annotate("grad = "+str(z[0].round(3))+" $\pm$ "+str(np.sqrt(cov[0][0]).round(3)),xy=(0,0.88*max_y))
ax3.annotate("int = "+str(z[1].round(3))+" $\pm$ "+str(np.sqrt(cov[1][1]).round(3)),xy=(0,0.78*max_y))
chi2vals = chisquare(xp,yp)
print(chi2vals)
ax3.annotate("Chi2 = "+str((chi2vals[0]).round(3))+", Redux = "+str((chi2vals[0]/(n_bins - 2)).round(3)),xy=(0,0.98*max_y))
ax3.set_title("Agreement")
ax3.set_xlabel(r"Initial $\eta$")
ax3.set_ylabel(r"Recon $\eta$")
f.savefig("./AEPerformance.png")

