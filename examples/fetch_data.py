from autodqm_ml.data_prep.data_fetcher import DataFetcher
from autodqm_ml.utils import setup_logger

logger = setup_logger("DEBUG", "output/log.txt")
fetcher = DataFetcher(
        tag = "test", # will identify output files
        contents = "metadata/cl2_1donly_example.json",
        datasets = "metadata/2017_dataset_example.json",
        short = False
)

fetcher.run()

