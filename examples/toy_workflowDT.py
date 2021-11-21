from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.algorithms.statistical_tester import StatisticalTester
from autodqm_ml.algorithms.autoencoder import AutoEncoder
from autodqm_ml.algorithms.pca import PCA
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--load_model', dest='load_model', type=bool, default=False, help='T/F load model')     
args = parser.parse_args()

from autodqm_ml.utils import setup_logger
logger = setup_logger("INFO")


training_file = 'scripts/output/test_SingleMuon.pkl' #"scripts/output/test_9Jun2021_SingleMuon.pkl"

wheels = [-2]#,0]
secs = [1,5,10]
sts = [1,2,3,4]

t0  = [f'DT/Run summary/02-Segments/Wheel{w}/Sector{sec}/Station{st}/T0_FromSegm_W{w}_Sec{sec}_St{st}' for w in wheels for sec in secs for st in sts]
#h4d = [f'DT/Run summary/02-Segments/Wheel{w}/Sector{sec}/Station{st}/h4DSegmNHits_W{w}_St{st}_Sec{sec}' for w,sec,st in zip(wheels,secs,sts)]
#vdrift = [f'DT/Run summary/02-Segments/Wheel{w}/Sector{sec}/Station{st}/VDrift_FromSegm_W{w}_Sec{sec}_St{st}' for w,sec,st in zip(wheels,secs,sts)]
histnames = t0 #+ h4d + vdrift 
histograms = {histname:{'normalize':True} for histname in histnames}

thresholds = {histname: 1 for histname in histnames}


p = PCA("my_pca")
a = AutoEncoder("my_autoencoder")
algos = [a]#,a]

for x in algos:
    x.load_data(
        file = training_file,
        histograms = histograms,
        train_frac = 0.5,
        remove_identical_bins = True,
        remove_low_stat = True
    )

    if args.load_model:
        x.load_model(model_file='models')
    else:
        x.train()
        x.save_model(model_file='models')
        

test_runs = algos[0].data["run_number"]["test"]
test = test_runs[0:10]
ref = test_runs[10]


results = {}
for x in algos:
    results[x.name] = x.evaluate(
            runs = test,
            reference = ref,
            histograms = list(histograms.keys()),
        thresholds = thresholds, 
    )
    
    x.plot(runs = test, 
           histograms = list(histograms.keys())
           )


for run in test:
    logger.info("Run: %d" % run)
    for x in algos:
        logger.info("Algorithm: %s, results: %s" % (x.name, results[x.name][run]))

