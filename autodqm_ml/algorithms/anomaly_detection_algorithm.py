import os
import pandas
import numpy
import awkward
import json
import csv

from autodqm_ml import utils
from autodqm_ml.data_formats.histogram import Histogram
from autodqm_ml.constants import kANOMALOUS, kGOOD
from autodqm_ml.rebinning import rebinning_min_occupancy_2d, rebinning_min_occupancy_1d
from autodqm_ml.evaluation import pull_tool

import logging
logger = logging.getLogger(__name__)

DEFAULT_COLUMNS = ["run_number", "label"] # columns which should always be read from input df

class AnomalyDetectionAlgorithm():
    """
    Abstract base class for any anomaly detection algorithm,
    including ks-test, pull-value test, pca, autoencoder, etc.
    :param name: name to identify this anomaly detection algorithm
    :type name: str
    """

    def __init__(self, name = "default", **kwargs):
        self.name = name

        self.data_is_loaded = False

        # These arguments will be overwritten if provided in kwargs
        self.output_dir = "output"
        #self.tag = ""
        #self.algorithm = ""
        self.histograms = {}
        self.input_file = None
        self.remove_low_stat = True
        self.integrals = None
        self.means = None

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
        

    def load_data(self, file = None, histograms = {}, train_frac = 0.5, remove_low_stat = True):
        """
        Loads data from pickle file into ML class. 

        :param file: file containing data to be extracted. File output of fetch_data.py
        :type file: str
        :param histograms: names of histograms to be loaded. Must match histogram names used in fetch_data.py. Dictionary in the form {<histogram name> : {"normalize" : <bool>}}.
        :type histograms: dict. Default histograms = {}
        :param train_frac: fraction of dataset to be kept as training data. Must be between 0 and 1. 
        :type train_frac: float. Default train_frac = 0.0
        :param remove_low_stat: removes runs containing histograms with low stats. Low stat threshold is 1000 events.
        :type remove_low_stat: bool. remove_low_stat = False
        """
        if self.data_is_loaded:
            return

        if file is not None:
            if self.input_file is not None:
                if not (file == self.input_file):
                    logger.warning("[AnomalyDetectionAlgorithm : load_data] Data file was previously set as '%s', but will be changed to '%s'." % (self.input_file, file)) 
                    self.input_file = file
            else:
                self.input_file = file

        if self.input_file is None:
            logger.exception("[AnomalyDetectionAlgorithm : load_data] No data file was provided to load_data and no data file was previously set for this instance, please specify the input data file.")
            raise ValueError()

        if not os.path.exists(self.input_file):
            self.input_file = utils.expand_path(self.input_file)

        if histograms:
            self.histograms = histograms
        self.histogram_name_map = {} # we replace "/" and spaces in input histogram names to play nicely with other packages, this map lets you go back and forth between them

        logger.debug("[AnomalyDetectionAlgorithm : load_data] Loading training data from file '%s'" % (self.input_file))

        # Load dataframe
        df = awkward.from_parquet(self.input_file)
       
        # Set helpful metadata
        for histogram, histogram_info in self.histograms.items():
            self.histograms[histogram]["name"] = histogram.replace("/", "").replace(" ","")
            self.histogram_name_map[self.histograms[histogram]["name"]] = histogram

            a = awkward.to_numpy(df[histogram][0])
            self.histograms[histogram]["shape"] = a.shape
            self.histograms[histogram]["n_dim"] = len(a.shape)
            self.histograms[histogram]["n_bins"] = 1
            for x in a.shape:
                self.histograms[histogram]["n_bins"] *= x 

        hist_integrals = []
        hist_means = []
        #print(self.histograms[histogram]["name"])
        for histogram, histogram_info in self.histograms.items():
            # Normalize (if specified in histograms dict)
            if "normalize" in histogram_info.keys():
                if histogram_info["normalize"]:
                    if histogram_info["n_dim"] == 2:
                        #print(len(df[histogram]),len(df[histogram][0]),len(df[histogram][0][0]))
                        logger.debug("[anomaly_detection_algorithm : load_data] Rebinning and normalising the 2D histogram '%s'" % histogram)
                        df[histogram], hist_integral = rebinning_min_occupancy_2d(df[histogram], 0.01)
                        new_shape = (len(df[histogram][0]),)
                        self.histograms[histogram]["shape"] = new_shape
                        self.histograms[histogram]["n_dim"] = len(new_shape)
                        self.histograms[histogram]["n_bins"] = len(df[histogram][0])
                        #print(len(new_shape))
                        hist_original = df[histogram]
                        mean_histogram = awkward.mean(df[histogram], axis=0)
                        df[histogram] = awkward.Array([arr - mean_histogram for arr in df[histogram]])

                        logger.debug("[anomaly_detection_algorithm : load_data] Now calculating the mean of this 2D histogram and subtracting this from each individual rebinned and normalised histogram '%s'" % histogram)
                    else:
                        sum = awkward.sum(df[histogram], axis = -1)
                        hist_integral = sum
                        #logger.debug("[anomaly_detection_algorithm : load_data] Normalising the 1D histogram '%s' by the sum of total entries." % histogram)
                        df[histogram] = df[histogram] * (1. / sum)
                        logger.debug("[anomaly_detection_algorithm : load_data] Rebinning and normalising the 1D histogram '%s'" % histogram)
                        df[histogram] = rebinning_min_occupancy_1d(df[histogram], 0.001)
                        mean_histogram = awkward.mean(df[histogram], axis=0)
                        df[histogram] = awkward.Array([arr - mean_histogram for arr in df[histogram]])
                        logger.debug("[anomaly_detection_algorithm : load_data] Now calculating the mean of this 1D histogram and subtracting this from each individual rebinned and normalised histogram '%s'" % histogram)
            hist_integrals.append(numpy.array(hist_integral).ravel())
            hist_means.append(mean_histogram)

        self.n_train = awkward.sum(df.label == 0)
        self.n_bad_runs = awkward.sum(df.label != 0)
        self.df = df
        self.n_histograms = len(list(self.histograms.keys()))
        self.integrals = hist_integrals
        self.means = hist_means

        logger.debug("[AnomalyDetectionAlgorithm : load_data] Loaded data for %d histograms with %d events in training set, excluding the %d bad runs." % (self.n_histograms, self.n_train, self.n_bad_runs))

        self.data_is_loaded = True


    def add_prediction(self, histogram, score, reconstructed_hist = None):
        """
        Add fields to the df containing the score for this algorithm (p-value/pull-value for statistical tests, sse for ML algorithms)
        and the reconstructed histograms (for ML algorithms only).
        """
        if reconstructed_hist is not None:
            self.df[histogram + "_reco_" + self.tag] = reconstructed_hist

        self.df[histogram + "_score_" + self.tag] = score * len(self.df[histogram][0])

    def save(self, histograms = {}, tag = "", algorithm = "", reco_assess_plots = False):
        """

        """
        os.system("mkdir -p %s" % self.output_dir)

        self.output_file = "%s/%s_%s_runs_and_sse_scores.csv" % (self.output_dir, self.input_file.split("/")[-1].replace(".parquet", ""), tag)

        if reco_assess_plots == True:
            output_parquet = "%s/%s.parquet" % (self.output_dir, self.input_file.split("/")[-1].replace(".parquet", ""))
            awkward.to_parquet(self.df, output_parquet)
            logger.info("[AnomalyDetectionAlgorithm : save] Saving output for plot assessment '%s'." % (output_parquet))

        chi2_analysis_output_file = "%s/%s_%s_chi2.csv" % (self.output_dir, self.input_file.split("/")[-1].replace(".parquet", ""), tag)
        chi2df = self.df

        reco_analysis_output_file = "%s/%s_%s_reco_integs.csv" % (self.output_dir, self.input_file.split("/")[-1].replace(".parquet", ""), tag)
        reco_df = self.df

        desired_hists_for_study = list(histograms.keys())
        score_columns = [hist_name + "_score_" + tag for hist_name in desired_hists_for_study]
        reco_columns = [hist_name + "_reco_" + tag for hist_name in desired_hists_for_study]
        columns_to_keep = ['run_number', 'year', 'label'] + score_columns
        chi2_cols_to_keep = ['run_number', 'year', 'label']
        reco_results = ['run_number', 'year', 'label']
        filtered_fields = {field: self.df[field] for field in self.df.fields if field in columns_to_keep}
        chi2_filtered_fds = {field: chi2df[field] for field in chi2df.fields if field in chi2_cols_to_keep}
        reco_fields = {field: reco_df[field] for field in reco_df.fields if field in reco_results}

        chi2_tol0_all_hists = []
        maxpull_tol0_all_hists = []
        chi2_tol1_all_hists = []
        maxpull_tol1_all_hists = []
        data_raw_all_hists = []
        ref_raw_all_hists = []

        for hist_iter in range(len(desired_hists_for_study)):
            data_raw = chi2df[desired_hists_for_study[hist_iter]] * self.integrals[hist_iter][:, numpy.newaxis]
            ref_raw = numpy.array(chi2df[reco_columns[hist_iter]] * 100*self.integrals[hist_iter][:, numpy.newaxis])
            ref_raw[ref_raw < 0] = 0
            ref_list_raw = numpy.array([[subarray] for subarray in ref_raw])
            run_list = self.df['run_number']

            chi2_tol0_vals = []
            maxpull_tol0_vals = []
            chi2_tol1_vals = []
            maxpull_tol1_vals = []
            data_hist_raw_vals = []
            ref_hist_raw_vals = []

            for run in range(len(data_raw)):
                data_hist_raw = numpy.round(numpy.copy(numpy.float64(data_raw[run])))
                ref_hists_raw = numpy.round(numpy.array([numpy.copy(numpy.float64(ref_list_raw[run]))]))
                nBinsUsed = numpy.count_nonzero(numpy.add(ref_hists_raw[0], data_hist_raw))
                [pull_hist, ref_hist_prob_wgt] = pull_tool.pull(data_hist_raw, ref_hists_raw, 0)
                chi2_tol0_AB = numpy.square(pull_hist).sum()/nBinsUsed
                maxpull_tol0_AB = pull_tool.maxPullNorm(numpy.amax(pull_hist), nBinsUsed)
                min_pull_tol0_AB = pull_tool.maxPullNorm(numpy.amin(pull_hist), nBinsUsed)
                if abs(min_pull_tol0_AB) > maxpull_tol0_AB: maxpull_tol0_AB = min_pull_tol0_AB
                [pull_hist, ref_hist_prob_wgt] = pull_tool.pull(data_hist_raw, ref_hists_raw, 0.01)
                chi2_tol1_AB = numpy.square(pull_hist).sum()/nBinsUsed
                maxpull_tol1_AB = pull_tool.maxPullNorm(numpy.amax(pull_hist), nBinsUsed)
                min_pull_tol1_AB = pull_tool.maxPullNorm(numpy.amin(pull_hist), nBinsUsed)
                if abs(min_pull_tol1_AB) > maxpull_tol1_AB: maxpull_tol1_AB = min_pull_tol1_AB
                chi2_tol0_vals.append(chi2_tol0_AB)
                maxpull_tol0_vals.append(maxpull_tol0_AB)
                chi2_tol1_vals.append(chi2_tol1_AB)
                maxpull_tol1_vals.append(maxpull_tol1_AB)
                data_hist_raw_vals.append(numpy.array(data_hist_raw))
                ref_hist_raw_vals.append(numpy.array(ref_hists_raw[0][0]))
                #if run < 3:
                #    print("data")
                #    print(data_hist_raw)
                #    print("ref")
                #    print(ref_hists_raw[0][0])
                
            chi2_tol0_all_hists.append(chi2_tol0_vals)
            maxpull_tol0_all_hists.append(maxpull_tol0_vals)
            chi2_tol1_all_hists.append(chi2_tol1_vals)
            maxpull_tol1_all_hists.append(maxpull_tol1_vals)
            data_raw_all_hists.append(numpy.array(data_raw))
            ref_raw_all_hists.append(numpy.array(ref_raw))

        self.df = awkward.zip(filtered_fields)
        chi2df = awkward.zip(chi2_filtered_fds)

        complete_set_of_integrals = []
        for hist_iter in range(len(reco_columns)):
            integ_info = reco_df[reco_columns[hist_iter]]
            integrals_for_each_run = []
            for run in range(len(integ_info)):
                integ_value = numpy.sum(numpy.copy(numpy.array(integ_info[run])))
                #if reco_df["run_number"][run] == 361365:
                #    print(reco_df["run_number"][run])
                #    print(desired_hists_for_study[hist_iter] + "," + str(integ_value))
                integrals_for_each_run.append(integ_value)
            complete_set_of_integrals.append(integrals_for_each_run)

        reco_df = awkward.zip(reco_fields)

        if algorithm.lower() in ["ae","autoencoder"]:
            algo_name = "ae"
        elif algorithm.lower() == "pca":
            algo_name = "pca"

        if algo_name is not None:
            algo_field = awkward.Array([algo_name] * len(self.df))
            self.df = awkward.with_field(self.df, algo_field, "algo")
            chi2df = awkward.with_field(chi2df, algo_field, "algo")
            reco_df = awkward.with_field(reco_df, algo_field, "algo")

        chi2_tol0_all_hists = numpy.array(chi2_tol0_all_hists)
        maxpull_tol0_all_hists = numpy.array(maxpull_tol0_all_hists)
        chi2_tol1_all_hists = numpy.array(chi2_tol1_all_hists)
        maxpull_tol1_all_hists = numpy.array(maxpull_tol1_all_hists)

        for hist_iter in range(len(desired_hists_for_study)):
            chi2_tol0_field = awkward.Array(chi2_tol0_all_hists[hist_iter])
            maxpull_tol0_field = awkward.Array(maxpull_tol0_all_hists[hist_iter])
            chi2_tol1_field = awkward.Array(chi2_tol1_all_hists[hist_iter])
            maxpull_tol1_field = awkward.Array(maxpull_tol1_all_hists[hist_iter])
            data_raw_field = awkward.Array(data_raw_all_hists[hist_iter])
            ref_raw_field = awkward.Array(ref_raw_all_hists[hist_iter])
            chi2df = awkward.with_field(chi2df, chi2_tol0_field, desired_hists_for_study[hist_iter] + "_chi2_tol0")
            chi2df = awkward.with_field(chi2df, maxpull_tol0_field, desired_hists_for_study[hist_iter] + "_maxpull_tol0")
            chi2df = awkward.with_field(chi2df, chi2_tol1_field, desired_hists_for_study[hist_iter] + "_chi2_tol1")
            chi2df = awkward.with_field(chi2df, maxpull_tol1_field, desired_hists_for_study[hist_iter] + "_maxpull_tol1")
            chi2df = awkward.with_field(chi2df, data_raw_field, desired_hists_for_study[hist_iter] + "_original")
            chi2df = awkward.with_field(chi2df, ref_raw_field, desired_hists_for_study[hist_iter] + "_prediction")
            chi2df = awkward.with_field(chi2df, self.integrals[hist_iter], desired_hists_for_study[hist_iter] + "_integral")

        complete_set_of_integrals = numpy.array(complete_set_of_integrals)
        for hist_iter in range(len(reco_columns)):
            complete_set_of_integrals_field = awkward.Array(complete_set_of_integrals[hist_iter])
            reco_df = awkward.with_field(reco_df, complete_set_of_integrals_field, reco_columns[hist_iter] + "_reco_integrals")

        list_of_dicts = awkward.to_list(self.df)
        chi2_dicts = awkward.to_list(chi2df)
        reco_df_dicts = awkward.to_list(reco_df)
        fieldnames = list_of_dicts[0].keys()
        chi2dfnames = chi2_dicts[0].keys()
        recodfnames = reco_df_dicts[0].keys()

        with open(self.output_file, 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(list_of_dicts)

        with open(chi2_analysis_output_file, 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.DictWriter(csv_file, fieldnames=chi2dfnames)
            csv_writer.writeheader()
            csv_writer.writerows(chi2_dicts)

        with open(reco_analysis_output_file, 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.DictWriter(csv_file, fieldnames=recodfnames)
            csv_writer.writeheader()
            csv_writer.writerows(reco_df_dicts)

        self.config_file = "%s/%s_%s.json" % (self.output_dir, self.name, self.tag)
        logger.info("[AnomalyDetectionAlgorithm : save] Saving output for large data SSE assessment '%s'." % (self.output_file))
        config = {}
        for k,v in vars(self).items():
            if utils.is_json_serializable(v):
                config[k] = v

        logger.info("[AnomalyDetectionAlgorithm : save] Saving AnomalyDetectionAlgorithm config to file '%s'." % (self.config_file))
        with open(self.config_file, "w") as f_out:
            json.dump(config, f_out, sort_keys = True, indent = 4)
