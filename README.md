# AutoDQM_ML
[![DOI](https://zenodo.org/badge/356313006.svg)](https://zenodo.org/badge/latestdoi/356313006)

## Description
This repository contains tools relevant for training and evaluating anomaly detection algorithms on CMS DQM data.
Core code is contained in `autodqm_ml`, core scripts are contained in `scripts` and some helpful examples are in `examples`.
See the README of each subdirectory for more information on each.

## Installation
**1. Clone repository**
```
git clone https://github.com/AutoDQM/AutoDQM_ML.git 
cd AutoDQM_ML
```
**2. Install dependencies**

Dependencies are listed in ```environment.yml``` and installed using `conda`. If you do not already have `conda` set up on your system, you can install (for linux) with:
```
curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
```
You can then set `conda` to be available upon login with
```
~/miniconda3/bin/conda init # adds conda setup to your ~/.bashrc, so relogin after executing this line
```

Once `conda` is installed and set up, install dependencies with (warning: this step may take a while)
```
conda env create -f environment.yml -p <path to install conda env>
```

Some packages cannot be installed via `conda` or take too long and need to be installed with `pip` (after activating your `conda` env above):
```
pip install yahist
pip install tensorflow==2.5
```

Note: if you are running on `lxplus`, you may run into permissions errors, which may be fixed with:
```
chmod 755 -R /afs/cern.ch/user/s/<your_user_name>/.conda
```
and then rerunning the command to create the `conda` env. The resulting `conda env` can also be several GB in size, so it may also be advisable to specify the installation location in your work area if running on `lxplus`, i.e. running the `conda env create` command with `-p /afs/cern.ch/work/...`.

**3. Install autodqm-ml**

Install with:
```
pip install -e .
```

Once your setup is installed, you can activate your python environment with
```
conda activate autodqm-ml
```

**Note**: `CMSSW` environments can interfere with `conda` environments. Recommended to unset your CMSSW environment (if any) by running
```
eval `scram unsetenv -sh`
```
before attempting installation and each time before activating the `conda` environment.

## Development Guidelines

### Documentation
Please comment code following [this convention](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) from `sphinx`.

In the future, `sphinx` can be used to automatically generate documentation pages for this project.

### Logging
Logging currently uses the Python [logging facility](https://docs.python.org/3/library/logging.html) together with [rich](https://github.com/willmcgugan/rich) (for pretty printing) to provide useful information printed both to the console and a log file (optional).

Two levels of information can be printed: `INFO` and `DEBUG`. `INFO` level displays a subset of the information printed by `DEBUG` level.

A logger can be created in your script with
```
from autodqm_ml.utils import setup_logger
logger = setup_logger(<level>, <log_file>)
```
And printouts can be added to the logger with:
```
logger.info(<message>) # printed out only in INFO level
logger.debug(<message>) # printed out in both INFO and DEBUG levels
```

It is only necessary to explicit create the logger with `setup_logger` once (likely in your main script). Submodules of `autodqm_ml` should initialize loggers as:
```
import logging
logger = logging.getLogger(__name__)
```
If a logger has been created in your main script with `setup_logger`, the line `logger = logging.getLogger(__name__)` will automatically detect the existing logger and inherit its settings (print-out level and log file).

Some good rules of thumb for logging:
```
logger.info # important & succint info that user should always see
logger.debug # less important info, or info that will have many lines of print-out
logger.warning # for something that may result in unintended behavior but isn't necessarily wrong
logger.exception # for something where the user definitely made a mistake
```

### Contributing
To contribute anything beyond a minor bug fix or modifying documentation/comments, first check out a new branch:
```
git checkout -b my_new_improvement
```
Add your changes to this branch and push:
```
git push origin my_new_improvement
```
Finally, when you think it's ready to be included in the main branch create a pull request (if you push your changes from the command line, Github should give you a link that you can click to automatically do this.) 

If you think the changes you are making might benefit from discussion, create an "Issue" under the [Issues](https://github.com/AutoDQM/AutoDQM_ML/issues) tab.
