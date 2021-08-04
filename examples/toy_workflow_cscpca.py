from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.algorithms.pca import PCA

from autodqm_ml.utils import setup_logger
logger = setup_logger("DEBUG")

training_file = "scripts/output/test_28Jul2021_SingleMuon.pkl"
histograms = {
        'CSC//Run summary/CSCOfflineMonitor/Digis/hWireTBin_p11b' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/Segments/hSnSegments' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/Segments/hSGlobalTheta' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/Segments/hSGlobalPhi' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/Segments/hSTimeCombined' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/recHits/hRHTimingAnodem11a' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/recHits/hRHTimingm22' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/recHits/hRHSumQm11a' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/recHits/hRHnrechits' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/Digis/hWireTBin_p32' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/BXMonitor/hCLCTL1A' : { "normalize" : True },
        'CSC//Run summary/CSCOfflineMonitor/Resolution/hSResidp12' : { "normalize" : True }        
        #'L1T//Run summary/L1TStage2EMTF/emtfTrackQualityVsMode' : { "normalize" : True },
}

p = PCA("my_pca")


for x in [p]:
    x.load_data(
            file = training_file,
            histograms = histograms,
            train_frac = 0.5
    )

    if isinstance(x, MLAlgorithm):
        x.train()


test_runs = p.data["run_number"]["test"]
test = test_runs[0:10]
ref = test_runs[10]

results = {}
for x in [p]:
    results[x.name] = x.evaluate(
            runs = test,
            reference = ref,
            histograms = ['CSC//Run summary/CSCOfflineMonitor/Digis/hWireTBin_p11b']
    )


for run in test:
    logger.info("Run: %d" % run)
    for x in [p]:
        logger.info("Algorithm: %s, results: %s" % (x.name, results[x.name][run]))

