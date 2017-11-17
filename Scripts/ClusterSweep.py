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
#SBATCH --gres={gpus}
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
echo $CDISCOUNT_DATASET

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
        "targetSize": (180, 180),
        "batchSize": 512,
        "epochs": 100,
        "maxValImages": 200000,
        "valTrainSplit": {
            "splitPercentage": 0.2,
            "dropoutPercentage": 0.0,
            "seed": 0
            },
        "model": {
            "name": "Xception",
            "kwargs": {"trainable": "onlyTop", "trainableFromBlock": 10}
            },
        "optimizer": {
            "name": "Adam",
            "kwargs": {"lr": 5e-3}
            },
        "epochSpecificParams":{
            1: {"lrDecayCoef": 0.1},
            2: {"lrDecayCoef": 0.1},
            3: {"lrDecayCoef": 0.1},
            4: {"lrDecayCoef": 0.1},
            }
        }
    
    # Resources
    resources = {
       "nodes": 1,
       "cpusPerTask": 5,
       "mem": "16G",
       "walltime": "8-00:00:00",
       "gpus": "gpu:tesla:1"
    }
    
    # Sweep params
    sweepParam = ("model", "kwargs", "trainable")
    sweepValues = ["blocks"]
    
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
        cmd = "python TrainModel.py \"%s\"" % (path.abspath(paramsFile))
        slurmScript = HPC_SCRIPT_TEMPLATE.format(jobName = friendlyName,
                                                 cmd = cmd,
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
        
        
            