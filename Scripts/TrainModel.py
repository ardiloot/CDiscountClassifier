import cProfile, pstats
from CDiscountClassifier import CDiscountClassfier

if __name__ == "__main__":
    params = {
        "datasetDir": None,
        "trainDatasetName": "train_example",
        "targetSize": (180, 180),
        "batchSize": 10 #64
        }
    
    params["valTrainSplit"] = {
        "splitPercentage": 0.2,
        "dropoutPercentage": 0.0,
        "seed": 0
        }
    
    params["model"] = {
        "name": "Xception",
        "kwargs": {}
        }
    
    params["fitGenerator"] = {
        "epochs": 5,
        "workers": 5
        }
    
    params["optimizer"] = {
        "name": "Adam",
        "kwargs": {
            "lr": 1e-3, 
            "beta_1": 0.9, 
            "beta_2": 0.999, 
            "epsilon": 1e-8, 
            "decay": 0.0
            }
        }

    #params["optimizer"] = {
    #    "name": "SGD",
    #    "kwargs": {
    #        "lr": 1e-2,
    #        "momentum": 0.0, 
    #        "decay": 0.0,
    #        "nesterov": False
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

    profile = cProfile.Profile()
    profile.enable()
    
    m = CDiscountClassfier(**params)
    m.InitTrainingData()
    m.TrainModel()
        
    profile.disable()
    pstats.Stats(profile).sort_stats("cumtime").print_stats(50)