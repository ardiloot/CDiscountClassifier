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
    print("Start TrainModel")
    sys.stdout.flush()
    
    params = {
        "datasetDir": None,
        "trainDatasetName": "train",
        #"resultsDir": r"C:\Users\Ardi\Downloads\results",
        "resultsDir": r"../../results",
        "targetSize": (180, 180),
        "batchSize": 32,
        "epochs": 3,
        "trainImagesPerEpoch": 100,
        "valImagesPerEpoch": 50,
        "testDropout": 0.9999,
        "valTrainSplit": {
            "splitPercentage": 0.2,
            "dropoutPercentage": 0.9999,
            "seed": 0
            },
        "trainAugmentation": {
            "zoom_range": 0.1,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "horizontal_flip": True,
            },
        "model": {
            "name": "Xception",
            "kwargs": {
                "trainable": "blocks",
                "trainableFromBlock": 10,
                "weights": "20171118-162839_Xception_trainAugmentation_nr_0/model.11-0.64.hdf5"
                },
            "trainMode": "continue",  
            },
        "optimizer": {
            "name": "Adam",
            "kwargs": {}
            },
        "epochSpecificParams":{
            2: {"lrDecayCoef": 0.1},
            4: {"lrDecayCoef": 0.1},
            6: {"lrDecayCoef": 0.1},
            }
        }

    if len(sys.argv) > 1:
        ymlParamsFile = sys.argv[1]
        print("ymlParamsFile", ymlParamsFile)
        with open(ymlParamsFile, "r") as fin:
            params = yaml.safe_load(fin)
    
    print("Init classifier...")
    sys.stdout.flush()
    m = CDiscountClassfier(**params)
    print("Init classifier done.")
    print("InitTrainingData...")
    sys.stdout.flush()
    m.InitTrainingData()
    print("InitTrainingData done.")
    sys.stdout.flush()
    
    m.GenerateTrainingName()
    print("Init tee...")
    sys.stdout.flush()
    tee = Tee(m.logFilename, "w")
    print("Init tee done.")
    sys.stdout.flush()
    
    m.TrainModel(updateTrainingName = False)
    
    profile = cProfile.Profile()    
    profile.enable()
    m.ValidateModel()
    profile.disable()
    pstats.Stats(profile).sort_stats("cumtime").print_stats(50)
    
    m.InitTestData()
    profile = cProfile.Profile()    
    profile.enable()
    m.PrepareSubmission()
    profile.disable()
    pstats.Stats(profile).sort_stats("cumtime").print_stats(50)
    
    
    del tee