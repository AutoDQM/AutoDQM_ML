import numpy as np 
from pathlib import Path
import awkward
import plotly.graph_objects as go


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

    mse = np.mean(np.square(original_hist - reconstructed_hist))
    
    # for bin edges
    binEdges = np.linspace(0, 1, original_hist.shape[0])
    width = binEdges[1] - binEdges[0]
    plotname = hist_path.split('/')[-1]
    text = '\n'.join((
        f'mse: {mse:.4e}',
        ))

    # creating figure and plotting the bar plots of original and reconstructed hists
    c = go.Figure() 
    c.add_trace(go.Bar(name="original", x=binEdges, y=original_hist, marker_color='white', marker=dict(line=dict(width=1,color='red'))))
    c.add_trace(go.Bar(name="reconstructed", x=binEdges, y=reconstructed_hist, marker_color='rgb(204, 188, 172)', opacity=.9))
    c['layout'].update(bargap=0)
    c['layout'].update(barmode='overlay')
    c['layout'].update(plot_bgcolor='white')
    c.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False)
    c.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False)
    c.update_layout(
        title=f'{plotname} {run} {algo}' , title_x=0.5,
        font=dict(
            family="Times New Roman",
            size=12,
            color="black"
        )
    )

    # add mse onto plots
    c['layout'].update(
        annotations=[
            dict(
                text= f'mse: {mse:.4e}',
                xref="x domain",
                yref="y domain",
                x = 0.96,
                y = 0.05,
                bordercolor='black',
                borderwidth=1,
                font = {'size':13},
                bgcolor = 'rgba(255,255,255,0.8)', 
                borderpad = 3, 
                showarrow=False
            )
        ]
    )
    Path(f'plots/{algo}/{run}').mkdir(parents=True, exist_ok=True)
    c.write_image(f'plots/{algo}/{run}/{plotname}.png')
    

    # create mse plots

    print(threshold)

    if mse > threshold:
        c = go.Figure(go.Bar(x=binEdges, y=np.square(original_hist - reconstructed_hist)))
        c.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False)
        c.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False)
        c.update_layout(
            title = f'MSE {plotname} {run}', title_x=0.5,
            font=dict(
                family="Times New Roman",
                size=12,
                color="black"
            )   
        )
        c.write_image(f'plots/{algo}/{run}/{plotname}-MSE.png')

def plotMSESummary(original_hists, reconstructed_hists, thresholds, hist_paths, runs, algo): 
    """ 
    Plots all the MSE on one plot that also shows how many passese threhold
    

    :param original_hist: original histogram to be plotted  
    :type original_hist: numpy array of shape (n, )
    :param reconstructed_hist: reconstructed histogram from the ML algorithm
    :type reconstructed_hist: numpy array of shape (n, )
    :param thresholds: threshold to determind histogram anomaly
    :type threshold: dict or None
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
    
    mse = np.mean(np.square(original_hists - reconstructed_hists), axis=1)


    ## make the thresholds dict into an array that matches the size of all plots coming in. original_hists can be 
    ## many plots with same name (but different runs). make threshold arr that same size as original_hists with corresponding
    ## threshold val
    thresholdsarr = np.full(len(original_hists), 0.00001)
    for i,x in enumerate(hist_paths):
        if x in thresholds:
            thresholdsarr[i] = thresholds[x]
    
    ## count number of good and bad histogrms
    num_good_hists = np.count_nonzero(mse < thresholdsarr)
    num_bad_hists = np.count_nonzero(mse > thresholdsarr)
    
    ## get names of top 5 highest mse histograms
    ## plot_names[argsort[-1]] should be the highest mse histogram
    sortIdx = np.argsort(mse)
    
    numHistText = [f'num good hists: {num_good_hists}', f'num bad hists: {num_bad_hists}']
    ## mse summary is for all plots in all runs, so need to make names accordingly
    hist_names = [hist_path.split('/')[-1] for hist_path in hist_paths]
    hist_names_runs = [f'{hist_name} ({run})' for run in runs for hist_name in hist_names]


    ## account for case less than 5 plots were tested
    maxidx = min(5, len(hist_names_runs))+1
    ## plot_name is the whole directory, so we only need the last for histname
    rankHistText = [f'{hist_names_runs[sortIdx[i]].split("/")[-1]}: {mse[sortIdx[i]]:.4e}' for i in range(-1, -maxidx, -1)]
    text = '<br>'.join(numHistText + rankHistText)

    # create plots
    c = go.Figure(go.Histogram(x=mse))
    c.update_layout(
        title = 'Summary of all MSE values', title_x=0.5,
        xaxis_title= 'MSE values',
        font=dict(
            family="Times New Roman",
            size=12,
            color="black"
        ),
        annotations = [dict(
            text=text,
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=1,
            y=0,
            
            bordercolor='black',
            borderwidth=1,
            bgcolor = 'rgba(255,255,255,0.5)'
            
        )
                   ] 
    )
    c.write_image(f'plots/{algo}/MSE_Summary.png')
