import os
import sys
import yaml
import cProfile, pstats
from os import path
from CDiscountClassifier import CDiscountClassfier

class Tee(object):
    def __init__(self, name, mode):
        if not path.isdir(path.dirname(name)):
            os.mkdir(path.dirname(name))
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
        
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

if __name__ == "__main__":
    params = {
        "datasetDir": None,
        "trainDatasetName": "train",
        "resultsDir": r"C:\Users\Ardi\Downloads\results",
        "targetSize": (90, 90),
        "batchSize": 64,
        "epochs": 3,
        "valTrainSplit": {
            "splitPercentage": 0.2,
            "dropoutPercentage": 0.9999,
            "seed": 0
            },
        "model": {
            "name": "Xception",
            "kwargs": {}
            },
        "optimizer": {
            "name": "Adam",
            "kwargs": {}
            },
        "epochSpecificParams":{
            1: {"lr": 1e-3},
            2: {"lr": 1e-4}
            }
        }

    if len(sys.argv) > 1:
        ymlParamsFile = sys.argv[1]
        print("ymlParamsFile", ymlParamsFile)
        with open(ymlParamsFile, "r") as fin:
            newParams = yaml.safe_load(fin)
        params.update(newParams)

    profile = cProfile.Profile()
    profile.enable()
    
    m = CDiscountClassfier(**params)
    m.InitTrainingData()
    m.GenerateTrainingName()
    
    tee = Tee(m.logFilename, "w")
    m.TrainModel(updateTrainingName = False)
        
    profile.disable()
    pstats.Stats(profile).sort_stats("cumtime").print_stats(50)
    
    del tee