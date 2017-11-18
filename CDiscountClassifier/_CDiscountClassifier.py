import os
import numpy as np
import pandas as pd
import keras
import yaml

from os import path
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator

from CDiscountClassifier import _Models
from CDiscountClassifier._Utils import PrecalcDatasetMetadata, BSONIterator, TrainTimeStatsCallback
from CDiscountClassifier._HelperFunctions import RepeatAndLabel  # @UnresolvedImport

#===============================================================================
# CDiscountClassfier
#===============================================================================

class CDiscountClassfier:
    
    def __init__(self, **kwargs):
        # Default params
        self.params = {
            "friendlyName": None,
            "datasetDir": None,
            "resultsDir": "../../results",
            "trainDatasetName": None,
            "targetSize": (180, 180),
            "batchSize": 64,
            "epochs": 5,
            "trainSeed": 1000003,
            "valImagesPerEpoch": None,
            "trainImagesPerEpoch": None,
            "trainAugmentation": {},
            }
        
        self.params["valTrainSplit"] = {
            "splitPercentage": 0.2,
            "dropoutPercentage": 0.0,
            "seed": 0
            }
        
        self.params["model"] = {
            "name": "Xception",
            "kwargs": {}
            }
        
        self.params["optimizer"] = {
            "name": "Adam",
            "kwargs": {}
            }
        self.params["epochSpecificParams"] = {}
                  
        # Update params
        self.params.update(**kwargs)
        
        # Check dataset dir
        if self.datasetDir is None and "CDISCOUNT_DATASET" in os.environ:
            self.datasetDir = os.environ["CDISCOUNT_DATASET"]
        
        # Resultsdir
        if not path.isdir(self.resultsDir):
            os.mkdir(self.resultsDir)
            
        # Precalc
        self._ReadCategoryTree()
        
    def GenerateTrainingName(self):  
        dateStr = datetime.now().strftime("%Y%m%d-%H%M%S")
        friendlyName = self.params["model"]["name"] if self.params["friendlyName"] is None else self.params["friendlyName"] 
        self.trainingName = "%s_%s" % (dateStr, friendlyName)
        return self.trainingName 
                
    def InitTrainingData(self):
        # Products metadata
        self.productsMetaDf = self._ReadProductsMetadata(self.trainDatasetName)
        
        # Split to train and val
        print("Making train/val splits...")
        self.trainMetaDf, self.valMetaDf = self._MakeTrainValSets(self.productsMetaDf, \
                                            **self.params["valTrainSplit"])
        print("Train", self.trainMetaDf.shape)
        print("Val", self.valMetaDf.shape)
        print("Making train/val splits done.")
                
    def TrainModel(self, updateTrainingName = True, newTrainingName = None):
        # Training name
        if updateTrainingName:
            self.trainingName = self.GenerateTrainingName() if newTrainingName is None else newTrainingName 
        
        print(self.trainingName)
        print("Training with params:")
        print(self.params)
        
        # Init
        params = self.params
        np.random.seed(params["trainSeed"])
        
        # Make dirs
        for folder in [self.resultsDir, self.trainingDir]:
            if not path.isdir(folder):
                os.mkdir(folder)

        # Save params
        with open(path.join(self.trainingDir, "parameters.yml"), "w") as fout:
            yaml.safe_dump(self.params, fout)

        # Image data generators 
        preproccesingFunc = _Models.PREPROCESS_FUNCS[params["model"]["name"]]
        self._trainImageDataGenerator = ImageDataGenerator(\
            preprocessing_function = preproccesingFunc, **params["trainAugmentation"])
        self._valImageDataGenerator = ImageDataGenerator(preprocessing_function = preproccesingFunc)

        # Iterators
        print("Prepare iterators...")
        bsonFile = path.join(self.datasetDir, "%s.bson" % (self.trainDatasetName))
        
        trainGenerator = BSONIterator(bsonFile, self.productsMetaDf, self.trainMetaDf, \
            self.nClasses, self._trainImageDataGenerator, targetSize = self.targetSize, \
            withLabels = True, batchSize = self.batchSize, shuffle = True)

        valGenerator = BSONIterator(bsonFile, self.productsMetaDf, self.valMetaDf, \
            self.nClasses, self._trainImageDataGenerator, targetSize = self.targetSize, \
            withLabels = True, batchSize = self.batchSize, shuffle = True, lock = trainGenerator.lock)
        print("Prepare iterators done.")

        # Model
        print("Preparing model...")
        modelClass = _Models.MODELS[params["model"]["name"]]
        model = modelClass(self.imageShape, self.nClasses, **params["model"]["kwargs"])
        model.summary()
        
        # Optimizer
        if params["optimizer"]["name"] == "Adam":
            optimizer = keras.optimizers.Adam(**params["optimizer"]["kwargs"])
        elif params["optimizer"]["name"] == "SGD":
            optimizer = keras.optimizers.SGD(**params["optimizer"]["kwargs"])
        elif params["optimizer"]["name"] == "RMSprop":
            optimizer = keras.optimizers.RMSprop(**params["optimizer"]["kwargs"])
        else:
            raise NotImplementedError()
        
        # Compile
        model.compile(metrics = ["accuracy"],
                      loss = "categorical_crossentropy",
                      optimizer = optimizer)
        print("Preparing model done.")
        
        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir = self.trainingDir, write_graph = False),
            keras.callbacks.ModelCheckpoint(self.modelFilename, monitor = "val_acc", verbose = 1, save_best_only = True),
            TrainTimeStatsCallback(self.statsFilename)
            ]
         
        # Steps per epoch
        stepsPerEpoch = max(1, self.trainMetaDf.shape[0] // self.batchSize)
        stepsPerValidation = max(1, self.valMetaDf.shape[0] // self.batchSize)

        if params["trainImagesPerEpoch"] is not None:
            stepsPerEpoch = self.params["trainImagesPerEpoch"] // self.batchSize
        
        if params["valImagesPerEpoch"] is not None:
            stepsPerValidation = self.params["valImagesPerEpoch"] // self.batchSize

        print("stepsPerEpoch", stepsPerEpoch, stepsPerEpoch * self.batchSize)
        print("stepsPerValidation", stepsPerValidation, stepsPerValidation * self.batchSize)
        
        # Fit model 
        print("Fitting model...")
        totalEpocs = params["epochs"]
        epochSpecificParams = params["epochSpecificParams"]
        for curEpoch in range(totalEpocs):
            # Set learning rate
            if curEpoch in epochSpecificParams:
                print("Update optimizer params:", epochSpecificParams[curEpoch])
                oldLr = keras.backend.get_value(model.optimizer.lr)
                
                if "lr" in epochSpecificParams[curEpoch] and "lrDecayCoef" in epochSpecificParams[curEpoch]:
                    raise ValueError("Only one (lr or lrDecayCoef) can be specified")
                
                for k, v in epochSpecificParams[curEpoch].items():
                    if k == "lr":
                        keras.backend.set_value(model.optimizer.lr, v)
                    elif k == "lrDecayCoef":
                        curLr = keras.backend.get_value(model.optimizer.lr)
                        keras.backend.set_value(model.optimizer.lr, v * curLr)
                    else:
                        raise ValueError("Unknown param %s" % (k))
                print("LR", oldLr, "->", keras.backend.get_value(model.optimizer.lr))
            
            print ("Start training for epoch %d/%d" % (curEpoch, totalEpocs))
            model.fit_generator(trainGenerator,
                steps_per_epoch = stepsPerEpoch,
                validation_data = valGenerator,
                validation_steps = stepsPerValidation,
                callbacks = callbacks,
                epochs = curEpoch + 1,
                workers = 5,
                initial_epoch = curEpoch)
            print("Fit generator done.")
            
        print("Model fit done.")
                

    def _ReadCategoryTree(self):
        # Read
        df = pd.read_csv(path.join(self.datasetDir, "category_names.csv"))
        df.columns = ["categoryId", "categoryLevel1", "categoryLevel2", "categoryLevel3"]
        df["classId"] = range(df.shape[0])
        
        # Save
        self._dfCategories = df
        
        # Build index (categoryId -> classId) and (classId -> catecoryId)
        self._mapCategoryToClass = dict(zip(df.categoryId, df.classId))
        self._mapClassToCategory = dict(zip(df.categoryId, df.classId))
        
        # Add not cat mapping
        self._mapCategoryToClass[-1] = -1
        self._mapClassToCategory[-1] = -1
        
    def _ReadProductsMetadata(self, datasetName):
        print("Loading metadata...")
        productsMetaFile = path.join(self.datasetDir, "%s_metadata.csv" % (datasetName))
        
        if not path.isfile(productsMetaFile):
            PrecalcDatasetMetadata(datasetName, self.datasetDir)
        productsMetaDf = pd.read_csv(productsMetaFile, index_col = "productId")
        productsMetaDf["classId"] = productsMetaDf.categoryId.map(self._mapCategoryToClass)
        print("Metadata loaded.")
        
        return productsMetaDf
        
    def _MakeTrainValSets(self, productsMetaDf, splitPercentage = 0.2, dropoutPercentage = 0.0, seed = 0):
        np.random.seed(seed)
        indicesByGroups = productsMetaDf.groupby("categoryId", sort = False).indices
        numImgsColumnNr = productsMetaDf.columns.get_loc("numImgs")
        
        resVal, resTrain = [], []
        for _, indices in indicesByGroups.items():
            # Randomly drop
            toKeep = round(len(indices) * (1.0 - dropoutPercentage))
            if toKeep == 0:
                continue
            elif toKeep < len(indices):
                indices = np.random.choice(indices, toKeep, replace = False)
            
            # Validation set
            validationSize = max(min(1, len(indices) - 1), round(len(indices) * splitPercentage))
            validationIndices = np.random.choice(indices, validationSize, replace = False)
            validationIndicesSet = set(validationIndices)
            validationProductIds = productsMetaDf.index[validationIndices]
            validationNumImages = productsMetaDf.iloc[validationIndices, numImgsColumnNr]
            resVal.append(RepeatAndLabel(validationProductIds.values, validationNumImages.values))
            
            # Train set
            trainIndices = np.array([i for i in indices if i not in validationIndicesSet])
            trainProductIds = productsMetaDf.index[trainIndices]
            trainNumImages = productsMetaDf.iloc[trainIndices, numImgsColumnNr]
            resTrain.append(RepeatAndLabel(trainProductIds.values, trainNumImages.values))
     
        resTrainDf = pd.DataFrame(np.concatenate(resTrain, axis = 0), columns = ["productId", "imgNr"])
        resValDf = pd.DataFrame(np.concatenate(resVal, axis = 0), columns = ["productId", "imgNr"])
        return resTrainDf, resValDf    

    @property
    def datasetDir(self):  # @DuplicatedSignature
        return self.params["datasetDir"]

    @datasetDir.setter
    def datasetDir(self, value):
        self.params["datasetDir"] = value

    @property
    def trainDatasetName(self):
        return self.params["trainDatasetName"]

    @property
    def targetSize(self):
        return self.params["targetSize"]

    @property
    def batchSize(self):
        return self.params["batchSize"]
    
    @property
    def nClasses(self):
        return self._dfCategories.shape[0]

    @property
    def imageShape(self):
        return tuple(tuple(self.targetSize) + (3,))

    @property
    def resultsDir(self):
        return self.params["resultsDir"]

    @property
    def trainingDir(self):
        return path.join(self.resultsDir, self.trainingName)

    @property
    def modelFilename(self):
        return path.join(self.trainingDir, "model.{epoch:02d}-{val_acc:.2f}.hdf5")
    
    @property
    def statsFilename(self):
        return path.join(self.trainingDir, "accuracy.csv")
    
    @property
    def logFilename(self):
        return path.join(self.trainingDir, "log.txt")

if __name__ == "__main__":
    pass
