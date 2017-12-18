"""This script is used to train, validate and test the classifier. This script
optionally takes one command line parameter: parameterFile. It could be used to
run the script with presave parameters save into YAML file.

"""

import os
import sys
import yaml

from os import path
from CDiscountClassifier import CDiscountClassfier

#===============================================================================
# Constants
#===============================================================================

DEFAULT_PARAMS = {
    "datasetDir": None,
    "trainDatasetName": "train",
    "interpolationSize": (210, 210),
    "interpolation": "bicubic",
    "targetSize": (180, 180),
    "batchSize": 64,
    "epochs": 1,
    "trainImagesPerEpoch": 200,
    "valImagesPerEpoch": 100,
    "predictMethod": "productActivations",
    "testDropout": 0.9999,
    "workers": 5,
    "nTtaAugmentation": 5,
    "valTrainSplit": {
        "splitPercentage": 0.3,
        "dropoutPercentage": 0.9999,
        "seed": 0
        },
    "trainAugmentation": {
        "cropMode": "random",
        "cropProbability": 0.5,
        "horizontal_flip": True,
        },
    "model": {
        "name": "Xception",
        "kwargs": {
            "trainable": "onlyTop",
            "gpus": 1,
            "dropout": 0.5,
            },
        "trainMode": "continue",  
        },
    "optimizer": {
        "name": "AdamAccum",
        "kwargs": {
            "lr": 0.005,
            "accum_iters": 4,
            },
        },
    "epochSpecificParams":{
        4: {"lrDecayCoef": 0.1},
        6: {"lrDecayCoef": 0.1},
        }
    }

#===============================================================================
# Helper classes
#===============================================================================

class Tee(object):
    """Helper class to duplicate standard error and output to log file. Idea
    and code from https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python

    Parameters
    ----------
    name : string
        The filename for the file to duplicate.
    mode : string
        File open mode.

    """
    
    def __init__(self, name, mode):
        if not path.isdir(path.dirname(name)):
            os.mkdir(path.dirname(name))
        self.file = open(name, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        
    def __del__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()
        
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        self.stderr.flush()

if __name__ == "__main__":
    # Load params
    params = DEFAULT_PARAMS
    if len(sys.argv) > 1:
        ymlParamsFile = sys.argv[1]
        with open(ymlParamsFile, "r") as fin:
            params = yaml.safe_load(fin)
    
    # Init classifier
    print("Init classifier...")
    m = CDiscountClassfier(**params)
    m.InitTrainingData()
    m.GenerateTrainingName()
    
    # Duplicate stdout and stderr to a file
    print("Init tee...")
    tee = Tee(m.logFilename, "w")
    
    # Train model
    m.TrainModel(updateTrainingName = False)
    
    # Validate model
    m.ValidateModel()
    
    # Load test data and make predictions 
    m.InitTestData()
    m.PrepareSubmission()
    