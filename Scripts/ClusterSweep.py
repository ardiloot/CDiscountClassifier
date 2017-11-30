import os
import yaml
import numpy as np
import subprocess
from datetime import datetime
from copy import deepcopy
from os import path

HPC_SCRIPT_TEMPLATE = \
r"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:{gpus}
#SBATCH -J {jobName}

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpusPerTask}
#SBATCH --mem={mem}

#SBATCH --time={walltime}

uname -a
cd /gpfs/hpchome/ardiloot/cdiscount/code/Scripts
pwd

source activate python_3.6.2
python --version

export CDISCOUNT_DATASET=/tmp/ardiloot/cdiscount/dataset
printenv

{cmd}
"""

def ReplaceValue(params, param, value, copy = False):
    if copy:
        res = deepcopy(params)
    else:
        res = params
        
    d = res
    for key in list(param)[:-1]:
        d = d[key]
    
    name = param[-1]  
    if not name in d:
        raise ValueError("Key does not exist: %s" % (param)) 
    d[name] = value

    return res

if __name__ == "__main__":
    # Folders
    curFileDir = path.dirname(path.realpath(__file__))
    clusterSaveDir = path.abspath(path.join(curFileDir, "../../cluster"))
    if not path.isdir(clusterSaveDir):
        os.mkdir(clusterSaveDir)
    
    # Model params
    params = {
        "datasetDir": None,
        "trainDatasetName": "train",
        "interpolationSize": (180, 180),
        "interpolation": "bicubic",
        "targetSize": (180, 180),
        "batchSize": 128,
        "epochs": 100,
        "valImagesPerEpoch": 50000,
        "trainImagesPerEpoch": 2000000,
        "predictMethod": "meanActivations",
        "testDropout": 0.0,
        "trainAugmentation": {
            "cropMode": "random",
            },
        "valTrainSplit": {
            "splitPercentage": 0.1,
            "dropoutPercentage": 0.0,
            "seed": 0
            },
        "model": {
            "name": "Xception",
            "kwargs": {
                "trainable": "blocks10+", 
                "gpus": 2,
                "dropout": None,
                },
            "trainMode": "continue",  
            },
        "optimizer": {
            "name": "AdamAccum",
            "kwargs": {"lr": 5e-3, "accum_iters": 12}
            },
        "epochSpecificParams":{
            8: {"lrDecayCoef": 0.2},
            12: {"lrDecayCoef": 0.25},
            }
        }
    
    # Resources
    resources = {
       "nodes": 1,
       "cpusPerTask": 5,
       "mem": "20G",
       "walltime": "1-00:00:00",
    }
    
    # Sweep params
    sweepParam = ("model", "kwargs", "dropout")
    sweepValues = [0.2, 0.5]
    
    for i, sweepValue in enumerate(sweepValues):
        print(i, sweepParam, sweepValue)
        
        # Update params
        newParams = ReplaceValue(params, sweepParam, sweepValue, copy = True)
        
        # Name
        if type(sweepValue) in [str, int, float]:
            sweepValueStr = str(sweepValue)
        else:
            sweepValueStr = "nr_%d" % (i)
        friendlyName = "%s_%s_%s" % (newParams["model"]["name"], "_".join(sweepParam), sweepValueStr)
        newParams["friendlyName"] = friendlyName
        
        # Filename
        dateStr = datetime.now().strftime("%Y%m%d-%H%M%S")
        randomInt = np.random.randint(0, 1000)
        filename = "%s.%s_%s" % (dateStr, randomInt, friendlyName)
        
        # Save params to yaml
        paramsFile = path.join(clusterSaveDir, "%s.yml" % (filename))
        with open(paramsFile, "w") as fout:
            yaml.safe_dump(newParams, fout)
            
        # Write slurm script
        slurmFile = path.join(clusterSaveDir, "%s.sh" % (filename))
        gpus = newParams["model"]["kwargs"]["gpus"] if "gpus" in params["model"]["kwargs"] else 1
        cmd = "python TrainModel.py \"%s\"" % (path.abspath(paramsFile))
        slurmScript = HPC_SCRIPT_TEMPLATE.format(jobName = friendlyName,
                                                 cmd = cmd,
                                                 gpus = gpus,
                                                 **resources)
        
        with open(slurmFile, "w") as fout:
            fout.write(slurmScript)
            
        # Add to queue
        print("Adding to slurm")
        cwd = path.abspath(path.join(curFileDir, os.pardir))
        print("cwd", cwd)
        subprocess.Popen("sbatch \"%s\"" % (path.abspath(slurmFile)),
                         shell = True,
                         cwd = cwd)
        
        
            