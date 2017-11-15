import sys
import yaml
import cProfile, pstats
from CDiscountClassifier import CDiscountClassfier

if __name__ == "__main__":
    params = {
        "datasetDir": None,
        "trainDatasetName": "train",
        "targetSize": (180, 180),
        "batchSize": 32,
        "valTrainSplit": {
            "splitPercentage": 0.2,
            "dropoutPercentage": 0.995,
            "seed": 0
            },
        "model": {
            "name": "Xception",
            "kwargs": {}
            },
        "fitGenerator": {
            "epochs": 3,
            "workers": 5
            },
        "optimizer": {
            "name": "SGD",
            "kwargs": {
                "lr": 1e-3,
                "momentum": 0.9, 
                "decay": 0.0,
                "nesterov": False
                }
            }
        }
      
    #params["optimizer"] = {
    #    "name": "Adam",
    #    "kwargs": {
    #        "lr": 1e-3, 
    #        "beta_1": 0.9, 
    #        "beta_2": 0.999, 
    #        "epsilon": 1e-8, 
    #        "decay": 0.0
    #        }
    #    }
          
    #params["optimizer"] = {
    #    "name": "RMSprop",
    #    "kwargs": {
    #        "lr": 1e-3,
    #        "rho": 0.9,
    #        "epsilon": 1e-8,
    #        "decay": 0.0
    #        }
    #    }

    if len(sys.argv) > 1:
        ymlParamsFile = sys.argv[1]
        print("ymlParamsFile", ymlParamsFile)
        with open(ymlParamsFile, "r") as fin:
            newParams = yaml.safe_load(fin)
        params.update(newParams)

    print("Params", params)

    profile = cProfile.Profile()
    profile.enable()
    
    m = CDiscountClassfier(**params)
    m.InitTrainingData()
    m.TrainModel()
        
    profile.disable()
    pstats.Stats(profile).sort_stats("cumtime").print_stats(50)