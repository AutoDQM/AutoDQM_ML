#### THIS IS THE VERSION WITH THE FORMAT FOR THE PAPER #####
### this is the version that makes the combined plots of beta-binom + pca ###
## e

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)
## !!!TODO::change this to take arguments of csv files instead of hard code xlsx file
N = 3
algos = ['pull', 'chi2', 'chi2prime']#, 'pca']


counts_g = {}
counts_b = {}

for algo in algos:
    if (algo == 'chi2') or (algo == 'pull'):
        df = pd.read_csv('HLT_l1tShift_8_REF/L1T_HLTPhysics.csv')
        df = df.sort_values(by='run_number')
        df_g = df[df['label'] != 1]
        df_b = df[df['label'] == 1]

        ## filter out non score columns
        df_g = df_g.filter(regex=f'{algo}_score')
        df_b = df_b.filter(regex=f'{algo}_score')

    if algo == 'pca' :
        df = pd.read_csv('rob_csv/HLTPhysics_PCA_131224_sse_scores.csv')
        df = df.sort_values(by='run_number')
        df_g = df[df['label'] != 1]
        df_b = df[df['label'] == 1]

        ## filter out non score columns:
        df_g = df_g.filter(regex='_score_PCA_131224')
        df_b = df_b.filter(regex='_score_PCA_131224')

    if algo == 'chi2prime' :
        df = pd.read_csv('rob_csv/HLTPhysics_PCA_131224_modified_chi2_values.csv')
        df = df.sort_values(by='run_number')
        df_g = df[df['label'] != 1]
        df_b = df[df['label'] == 1]


        ## filter out non score columns:
        df_g = df_g.filter(regex=f'_{algo}')
        df_b = df_b.filter(regex=f'_{algo}')


    ## sort descending
    sorted_df_g = -np.sort(-df_g, axis=0)

    ## calculate thresholds
    cuts = np.array([(col[1:] + col[:-1])/2 for col in sorted_df_g.T]).T

    zerothcut = sorted_df_g[0,:] + (sorted_df_g[0, :] - sorted_df_g[1,:])/2
    cuts = np.insert(cuts, 0, zerothcut, axis=0)

    ## get counts
    counts_g[algo] = np.array([np.count_nonzero(df_g >= cut, axis=1) for cut in cuts])
    counts_b[algo] = np.array([np.count_nonzero(df_b >= cut, axis=1) for cut in cuts])



metrics_list =  [['pull','chi2'], ['chi2prime'], algos]
colors =  [ "#f89c20","#e42536", "#5790fc",]
markers = [ '-o','-D', '-^']
legendlabels =  ['Beta-binomial $\chi^2$ and max pull', 'PCA modified $\chi^2$', 'PCA and beta-binomial']

fig0, ax0= plt.subplots(figsize=(6,6))
fig1, ax1 = plt.subplots(figsize=(6,6))
for metrics, color, marker, legendlabel in zip(metrics_list, colors, markers, legendlabels):

    total_counts_g = np.zeros_like(counts_g['pull']) ## array of zero so we can append to it
    total_counts_b = np.zeros_like(counts_b['pull'])
    for metric in metrics:
        total_counts_g = total_counts_g + counts_g[metric]
        total_counts_b = total_counts_b + counts_b[metric]


    ## for HF, add all of the counts together by threshold before taking mean
    # avgcomb = (counts_g['pull'] + counts_g['chi2']).mean(axis=1)
    avg_cnt_g = total_counts_g.mean(axis=1)
    avg_cnt_b = total_counts_b.mean(axis=1)

    ## for RF, add all of the counts together by threshold before checking if greater than N, the divided by one of the counts.shape [1]
    # perccomb = np.count_nonzero((counts_g['pull'] + counts_g['chi2']) > N, axis=1)/counts_g['pull'].shape[1]
    perc_g = np.count_nonzero(total_counts_g >= N, axis=1)/total_counts_g.shape[1]
    perc_b = np.count_nonzero(total_counts_b >= N, axis=1)/total_counts_b.shape[1]


    ##--------- plotting the output in the same way as rob --------------
    # algorithm_name = "combined"
    # algorithm_name = # "chi2" if algo=='chi2' else "max pull"
    xaxislabelsize = 14
    yaxislabelsize = xaxislabelsize
    cmssize = 18
    luminositysize = 14
    axisnumbersize = 12
    annotatesize=16
    labelpad=10
    linewidth=1.5
    #legendlabel = f'{numref[0]} reference runs' if numref[0] != '1' else f'{numref[0]} reference run'
    ax1.tick_params(axis='both', which='major', labelsize=axisnumbersize)
    ax1.set_xlabel(f'Fraction of good runs with ≥{N} histogram flags', fontsize=xaxislabelsize, labelpad=labelpad)
    ax1.set_ylabel(f'Fraction of bad runs with ≥{N} histogram flags', fontsize=yaxislabelsize, labelpad=labelpad)
    ax1.axline((0, 0), slope=1, linestyle='--', linewidth=linewidth, color='#964a8b', zorder=0)
    ax1.plot(perc_g, perc_b, marker, mfc=color, color=color, mec='k', markersize=8, linewidth=1, label=legendlabel)
    ax1.axis(xmin=0,xmax=0.35,ymin=0,ymax=0.8)
    ax1.annotate(f"Combined beta-binomial and PCA tests", xy=(0.05, 0.98), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=14, fontstyle='italic')
    ax1.legend(loc='lower right', fontsize=annotatesize)
    ax1.text(0, 1.02, "CMS", fontsize=cmssize, weight='bold',transform=ax1.transAxes)
    ax1.text(0.15, 1.02, "Preliminary", fontsize=cmssize-4, fontstyle='italic', transform=ax1.transAxes)
    #ax1.text(0, 1.02, "Private work (CMS data)", fontsize=14, fontstyle='italic', transform=ax1.transAxes)
    ax1.text(0.63, 1.02, "2022 (13.6 TeV)", fontsize=14, transform=ax1.transAxes)

    ax0.tick_params(axis='both', which='major', labelsize=axisnumbersize)
    ax0.set_xlabel('Mean histogram flags per good run', fontsize=xaxislabelsize, labelpad=labelpad)
    ax0.set_ylabel('Mean histogram flags per bad run', fontsize=yaxislabelsize, labelpad=labelpad)
    ax0.axline((0, 0), slope=1, linestyle='--',linewidth=linewidth,color='#964a8b', zorder=0)
    ax0.plot(avg_cnt_g, avg_cnt_b, marker, mfc=color, color=color, mec='k', markersize=8, linewidth=1, label=legendlabel)
    ax0.annotate(f"Combined beta-binomial and PCA tests", xy=(0.05, 0.98), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=annotatesize-2, fontstyle='italic')
    ax0.axis(xmin=0,xmax=5,ymin=0,ymax=25) #20)
    ax0.legend(loc='center left', fontsize=annotatesize, bbox_to_anchor=(0.05, 0.75), bbox_transform=ax0.transAxes)
    ax0.text(0, 1.02, "CMS", fontsize=cmssize, weight='bold', transform=ax0.transAxes)
    ax0.text(0.15, 1.02, "Preliminary", fontsize=cmssize-4, fontstyle='italic', transform=ax0.transAxes)
    # ax0.text(0, 1.02, "Private work (CMS data)", fontsize=cmssize-4, fontstyle='italic', transform=ax0.transAxes)
    ax0.text(0.63, 1.02, "2022 (13.6 TeV)", fontsize=luminositysize, transform=ax0.transAxes)
    ## --------------------------------------------------------------------

    #for cut in cuts[:4]:
    #    print(cut)
fig0.savefig("plots/sorted/Combined_HF_ROC_comparison.pdf",bbox_inches='tight')
#print("SAVED: " + args.output_dir + "/RF_HF_ROC_comparison_" + algorithm_name + ".pdf")
fig1.savefig("plots/sorted/Combined_RF_ROC_comparison.pdf",bbox_inches='tight')
