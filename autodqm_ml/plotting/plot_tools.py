import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path
import awkward

from yahist import Hist1D

import logging
logger = logging.getLogger(__name__)

def make_sse_plot(name, recos, save_name, **kwargs):
    x_label = "Anomaly Score"
    y_label = "Fraction of runs"
    log_y = kwargs.get("log_y", False)

    # Append all sse's together to get a common binning
    all = awkward.concatenate([x["score"] for k, x in recos.items()])
    h_all = Hist1D(all)
    bins = h_all.edges
    
    hists = []

    for reco, info in recos.items():
        mean = awkward.mean(info["score"])
        std = awkward.std(info["score"])
        h = Hist1D(info["score"], bins = bins, label = reco + " [N = %d, SSE = %.2E +/- %.2E]" % (len(info["score"]), mean, std))
        h = h.normalize()
        hists.append(h)

    fig, ax = plt.subplots(1, figsize=(8,6))

    for idx, h in enumerate(hists):
        h.plot(ax=ax, color = "C%d" % (idx+1), errors = True, linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    ax.set_yscale("log")

    logger.debug("[plot_tools.py : make_sse_plot] Writing plot to file '%s'." % (save_name))
    plt.savefig(save_name)
    plt.savefig(save_name.replace(".pdf", ".png"))
    plt.clf()


def make_original_vs_reconstructed_plot(name, original, recos, run, save_name, **kwargs): 
    n_dim = len(np.array(original).shape)

    if n_dim == 1:
        make_original_vs_reconstructed_plot1d(name, original, recos, run, save_name, **kwargs)

    elif n_dim == 2:
        original_flat = awkward.flatten(original, axis = -1)
        recos_flat = {}
        for algorithm, reco in recos.items():
            recos_flat[algorithm] = {
                    "reco" : awkward.flatten(reco["reco"], axis = -1),
                    "score" : reco["score"]
            }
      
        make_original_vs_reconstructed_plot1d(name, original_flat, recos_flat, run, save_name, **kwargs)

    else:
        message = "[plot_tools.py : make_original_vs_reconstructed_plot] Plotting not implemented for histograms with dimension %d." % (n_dim)
        logger.exception(message)
        raise RuntimeError()

def make_original_vs_reconstructed_plot1d(name, original, recos, run, save_name, **kwargs):
    bins = "%s, 0, 1" % (len(original))
    x_label = name + " (a.u.)"
    y_label = "Fraction of events"
    
    rat_lim = kwargs.get("rat_lim", [0.0, 2.0])
    log_y = kwargs.get("log_y", False)

    h_orig = Hist1D(original, bins = bins, label = "original")
    h_orig._counts = original
    h_reco = []
    bbc = {reco: [] for reco in recos.keys()}
    nfp = {reco: [] for reco in recos.keys()}
    print("Run: "+str(run))
    for reco, info in recos.items():
        h = Hist1D(info["reco"], bins = bins, label = "%s [sse : %.2E]" % (reco, info["score"]))
        h._counts = info["reco"]
        h_reco.append(h)
        count_bad_bin = np.maximum(np.abs(original - info["reco"]),0.001)
        bad_bin_size = (np.asarray(count_bad_bin) > 0.001).sum()
        print("ALGORITHM: "+reco)
        if bad_bin_size > 0.1*len(info["reco"]):
            #print("ANOMALY IN BBC: Bad bin count greater than 10 ("+str(bad_bin_size)+") in run "+str(run))
            print("BBC: "+str(bad_bin_size))
        bbc[reco] = bad_bin_size
 
        reconozero = np.asarray(info["reco"])
        reconozero[reconozero < 0] = 0
        nfp_vals = np.asarray((original - reconozero)**2)
        for i in range(len(original)):
            if reconozero[i]-0.01*reconozero[i] < original[i] < reconozero[i]+0.01*reconozero[i]:
                nfp_vals[i] = 0
        nfp_tot = nfp_vals.sum()
        if nfp_tot > 1e-4:
            #print("ANOMALY IN 95% DEVIATION: SSE counted at the 99% percentile greater than 1e-6 ("+str(nfp_tot)+") in run "+str(run))
            print("OPD: "+str(nfp_tot))
        nfp[reco] = nfp_tot

    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(8,6), gridspec_kw=dict(height_ratios=[3, 1]))
    plt.grid()

    h_orig.plot(ax=ax1, color="black", errors = False, linewidth=2)
    plt.sca(ax1)

    for idx, h in enumerate(h_reco):
        h.plot(ax=ax1, color = "C%d" % (idx+1), errors = False, linewidth=2)

    for idx, h in enumerate(h_reco):
        ratio = h.divide(h_orig)
        ratio.metadata["label"] = None
        ratio.plot(ax=ax2, color = "C%d" % (idx+1), errors = False, linewidth=2)

    ax1.set_ylabel(y_label)
    ax2.set_ylabel("ML Reco / Original")
    ax2.set_xlabel(x_label)
    ax2.set_ylim(rat_lim)
    ax1.set_ylim([0.0, awkward.max(original) * 1.5])

    if log_y:
        ax1.set_yscale("log")

    #logger.debug("[plot_tools.py : make_original_vs_reconstructed_plot1d] Writing plot to file '%s'. " % (save_name))
    plt.savefig(save_name)
    plt.savefig(save_name.replace(".pdf", ".png"))
    plt.clf()


def plot1D(original_hist, reconstructed_hist, run, hist_path, algo, threshold):    
    """
    plots given original and recontructed histogram. Will plot the MSE plot if the SSE is over the threshold. 

    :param original_hist: original histogram to be plotted  
    :type original_hist: numpy array of shape (n, )
    :param reconstructed_hist: reconstructed histogram from the ML algorithm
    :type reconstructed_hist: numpy array of shape (n, )
    :param hist_path: name of histogram
    :type hist_path: str
    :param algo: name of algorithm used. This is used to label the folder. Can use self.name to be consistent between this and plotMSESummary
    :param type: str
    :param threshold: threshold to determind histogram anomaly
    :type threshold: int
    """
    fig, ax = plt.subplots()
    mse = np.mean(np.square(original_hist - reconstructed_hist))
    
    # for bin edges
    binEdges = np.linspace(0, 1, original_hist.shape[0])
    width = binEdges[1] - binEdges[0]
    # plot original/recon 
    ax.bar(binEdges, original_hist, alpha=0.5, label='original', width=width)
    ax.bar(binEdges, reconstructed_hist, alpha=0.5, label='reconstructed', width=width)

    plotname = hist_path.split('/')[-1]
    ax.set_title(f'{plotname} {run} {algo}')
    leg = ax.legend(loc='upper right')
    text = '\n'.join((
        f'mse: {mse:.4e}',
        ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(.8, 0.1*max(original_hist.max(), reconstructed_hist.max()), text, wrap=True, bbox=props )
    # create directory to save plot
    Path(f'plots/{algo}/{run}').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'plots/{algo}/{run}/{plotname}.png')
    plt.close('all')
    
    if mse > threshold: 
        fig2, ax2 = plt.subplots()
        ax2.bar(binEdges, np.square(original_hist - reconstructed_hist), alpha=0.5, width=width)
        ax2.set_title(f'MSE {plotname} {run}')
        fig2.savefig(f'plots/{algo}/{run}/{plotname}-MSE.png')
        plt.close('all')
    

def plotMSESummary(original_hists, reconstructed_hists, threshold, hist_paths, runs, algo): 
    """ 
    Plots all the MSE on one plot that also shows how many passese threhold
    

    :param original_hist: original histogram to be plotted  
    :type original_hist: numpy array of shape (n, )
    :param reconstructed_hist: reconstructed histogram from the ML algorithm
    :type reconstructed_hist: numpy array of shape (n, )
    :param threshold: threshold to determind histogram anomaly
    :type threshold: int
    :param hist_path: list of name of histograms
    :type hist_path: list
    :param runs: list of runs used for testing. Must be same as list passed into the pca or autoencoder.plot function 
    :param type: list
    :param algo: name of algorithm used. This is used to place the plot in the correct folder. Can use self.name to be consistent between this and plot1D
    :param type: str

    """ 
    ## convert to awkward array as not all hists have the same length
    original_hists = awkward.Array(original_hists)
    reconstructed_hists = awkward.Array(reconstructed_hists)

    fig, ax = plt.subplots()
    mse = np.mean(np.square(original_hists - reconstructed_hists), axis=1)
    
    ## count number of good and bad histogrms
    num_good_hists = np.count_nonzero(mse < threshold)
    num_bad_hists = np.count_nonzero(mse > threshold)
    
    ## get names of top 5 highest mse histograms
    ## plot_names[argsort[-1]] should be the highest mse histogram
    sortIdx = np.argsort(mse)

    hist,_, _ = ax.hist(mse)
    ax.set_xlabel('MSE values') 
    ax.set_title('Summary of all MSE values')
    
    numHistText = [f'num good hists: {num_good_hists}', f'num bad hists: {num_bad_hists}']
    ## mse summary is for all plots in all runs, so need to make names accordingly
    hist_names = [hist_path.split('/')[-1] for hist_path in hist_paths]
    hist_names_runs = [f'{hist_name} ({run})' for run in runs for hist_name in hist_names]


    ## account for case less than 5 plots were tested
    maxidx = min(5, len(hist_names_runs))+1
    ## plot_name is the whole directory, so we only need the last for histname
    rankHistText = [f'{hist_names_runs[sortIdx[i]].split("/")[-1]}: {mse[sortIdx[i]]:.4e}' for i in range(-1, -maxidx, -1)]
    text = '\n'.join(
        numHistText + rankHistText
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(1.1*max(mse), 0.5*max(hist), text, wrap=True, bbox=props)
    fig.savefig(f'plots/{algo}/MSE_Summary.png', bbox_inches='tight')


def plot_roc_curve(h_name, results, save_name, **kwargs):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both')
    ax1.grid(True)

    log = kwargs.get("log", False)

    idx = 0
    for algo, res in results.items():
        ax1.plot(
                res["fpr"],
                res["tpr"],
                color = "C%d" % (idx+1),
                label = algo + " [AUC = %.3f +/- %.3f]" % (res["auc"], res["auc_unc"])
        )
        ax1.fill_between(
                res["fpr"],
                res["tpr"] - (res["tpr_unc"] / 2.),
                res["tpr"] + (res["tpr_unc"] / 2.),
                color = "C%d" % (idx+1),
                alpha = 0.25
        )
        idx += 1

    if not log:
        plt.ylim(0,1)
        plt.xlim(0,1)
    else:
        plt.xlim(0.005, 1)
        plt.ylim(0,1)
        ax1.set_xscale("log")

    plt.xlabel("False Anomaly Rate (FPR)")
    plt.ylabel("Anomaly Detection Efficiency (TPR)")

    legend = ax1.legend(loc='lower right')

    logger.debug("[plot_tools.py : plot_roc_curve] Writing plot to file '%s'." % (save_name))
    plt.savefig(save_name)
    plt.savefig(save_name.replace(".pdf", ".png"))
    plt.clf()

