import os
import bson
import numpy as np
import pandas as pd
import keras
import yaml
import keras.backend as K

from os import path
from datetime import datetime

from CDiscountClassifier import _Models
from CDiscountClassifier._Utils import PrecalcDatasetMetadata, BSONIterator, \
    TrainTimeStatsCallback, SetEpochParams, MultiGPUModelCheckpoint, \
    CropImageDataGenerator, SGDAccum
from CDiscountClassifier._HelperFunctions import RepeatAndLabel  # @UnresolvedImport
import bson

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
            "testDatasetName": "test",
            "interpolationSize": (180, 180),
            "targetSize": (180, 180),
            "batchSize": 64,
            "epochs": 5,
            "trainSeed": 1000003,
            "valImagesPerEpoch": None,
            "trainImagesPerEpoch": None,
            "trainAugmentation": {},
            "valAugmentation": {},
            "testAugmentation": {},
            "predictMethod": "meanActivations",
            "testDropout": 0.0,
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
        self.trainProductsMetaDf = self._ReadProductsMetadata(self.trainDatasetName)
        
        # Split to train and val
        print("Making train/val splits...")
        self.trainMetaDf, self.valMetaDf = self._MakeTrainValSets(self.trainProductsMetaDf, \
                                            **self.params["valTrainSplit"])
        print("Train", self.trainMetaDf.shape)
        print("Val", self.valMetaDf.shape)
        print("Making train/val splits done.")
        
        # Image data generators 
        print("Init iterators...")
        preproccesingFunc = _Models.PREPROCESS_FUNCS[self.params["model"]["name"]]
        trainImageDataGenerator = CropImageDataGenerator(targetSize = self.targetSize,
            preprocessing_function = preproccesingFunc, **self.params["trainAugmentation"])
        valImageDataGenerator = CropImageDataGenerator(targetSize = self.targetSize, \
            preprocessing_function = preproccesingFunc, **self.params["valAugmentation"])

        # Iterators
        bsonFile = path.join(self.datasetDir, "%s.bson" % (self.trainDatasetName))
        
        self.trainGenerator = BSONIterator(bsonFile, self.trainProductsMetaDf, self.trainMetaDf, \
            self.nClasses, trainImageDataGenerator, interpolationSize = self.interpolationSize, \
            withLabels = True, batchSize = self.batchSize, shuffle = True)

        self.valGenerator = BSONIterator(bsonFile, self.trainProductsMetaDf, self.valMetaDf, \
            self.nClasses, valImageDataGenerator, interpolationSize = self.interpolationSize, \
            withLabels = True, batchSize = self.batchSize, shuffle = True, lock = self.trainGenerator.lock)
        
        print("Init iterators done.")
   
    def InitTestData(self):
        print("Init test data ...")
        # Products metadata
        self.testProductsMetaDf = self._ReadProductsMetadata(self.testDatasetName)
        self.testMetaDf, _ = self._MakeTrainValSets(self.testProductsMetaDf, \
            splitPercentage = 0.0, dropoutPercentage = self.params["testDropout"])
        print("Test", self.testProductsMetaDf.shape, self.testMetaDf.shape)
        print("Init iterators...")
        bsonFile = path.join(self.datasetDir, "%s.bson" % (self.testDatasetName))
        preproccesingFunc = _Models.PREPROCESS_FUNCS[self.params["model"]["name"]]
        
        testImageDataGenerator = CropImageDataGenerator(targetSize = self.targetSize, \
            preprocessing_function = preproccesingFunc, **self.params["testAugmentation"])
        
        self.testGenerator = BSONIterator(bsonFile, self.testProductsMetaDf, self.testMetaDf, \
            self.nClasses, testImageDataGenerator, interpolationSize = self.interpolationSize, \
            withLabels = False, batchSize = self.batchSize, shuffle = False)
        print("Init iterators done.")
        print("Init test data done.")
                
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

        # Model
        print("Preparing model...")
        model = _Models.GetModel(self.imageShape, self.nClasses, \
            weightsDir = self.resultsDir, name = params["model"]["name"], \
            **params["model"]["kwargs"])
        model.summary()
        
        # Train mode
        trainMode = params["model"]["trainMode"] if "trainMode" in params["model"] else "continue"
        print("epochsCompleted", model.epochsCompleted, trainMode)
        startEpoch = 0
        if trainMode == "restart":
            startEpoch = 0
        elif trainMode == "continue":
            startEpoch = model.epochsCompleted
        else:
            raise ValueError("Unkonwn trainMode", trainMode)
        
        # Optimizer
        if params["optimizer"]["name"] == "Adam":
            optimizer = keras.optimizers.Adam(**params["optimizer"]["kwargs"])
        elif params["optimizer"]["name"] == "SGD":
            optimizer = keras.optimizers.SGD(**params["optimizer"]["kwargs"])
        elif params["optimizer"]["name"] == "RMSprop":
            optimizer = keras.optimizers.RMSprop(**params["optimizer"]["kwargs"])
        elif params["optimizer"]["name"] == "SGDAccum":
            optimizer = SGDAccum(**params["optimizer"]["kwargs"])
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
            MultiGPUModelCheckpoint(self.modelFilename, monitor = "val_acc", verbose = 1, save_best_only = True),
            TrainTimeStatsCallback(self.statsFilename)
            ]
         
        # Steps per epoch
        stepsPerEpoch = max(1, self.trainMetaDf.shape[0] // self.batchSize)
        stepsPerValidation = max(1, self.valMetaDf.shape[0] // self.batchSize)

        if params["trainImagesPerEpoch"] is not None:
            stepsPerEpoch = max(1, self.params["trainImagesPerEpoch"] // self.batchSize)
        
        if params["valImagesPerEpoch"] is not None:
            stepsPerValidation = max(1, self.params["valImagesPerEpoch"] // self.batchSize)

        print("stepsPerEpoch", stepsPerEpoch, stepsPerEpoch * self.batchSize)
        print("stepsPerValidation", stepsPerValidation, stepsPerValidation * self.batchSize)
        
        # Fit model 
        print("Fitting model...")
        totalEpocs = params["epochs"]
        epochSpecificParams = params["epochSpecificParams"]
        
        # Set epoch params
        for curEpoch in range(0, startEpoch):
            SetEpochParams(model, curEpoch, epochSpecificParams) 
        
        if startEpoch >= totalEpocs:
            print("Model already fitted.")
        
        # Iterate over epochs
        for curEpoch in range(startEpoch, totalEpocs):
            # Set learning rate
            SetEpochParams(model, curEpoch, epochSpecificParams)
            print ("Start training for epoch %d/%d" % (curEpoch, totalEpocs))
            print("Optimizer params", model.optimizer.get_config())
            
            model.fit_generator(self.trainGenerator,
                steps_per_epoch = stepsPerEpoch,
                validation_data = self.valGenerator,
                validation_steps = stepsPerValidation,
                callbacks = callbacks,
                epochs = curEpoch + 1,
                workers = 5,
                initial_epoch = curEpoch)
            print("Fit generator done.")
            
        self.model = model
        print("Model fit done.")
                
    def Predict(self, bsonIterator, evaluate = False, topK = 0):
        print("Predict")
        
        predictMethods = {
            "meanActivations": lambda x: np.mean(x, axis = 0),
            "productActivations": lambda x: np.prod(x, axis = 0),
            "rmsActivations": lambda x: np.mean(x ** 2.0, axis = 0),
            "firstImage": lambda x: x[0, :],
            }
        
        finalPredictMethod = self.params["predictMethod"]
        if finalPredictMethod not in predictMethods:
            raise ValueError("Unknown predictMethod" , finalPredictMethod)
        
        GetActivations = lambda x: K.function([K.learning_phase()] + self.model.inputs, self.model.outputs)([0, x])[0]
       
        res = []
        resTopK = []
        correctPredictions = dict((k, 0) for k in predictMethods)
        imagesProcessed = 0
        totalPredictions = 0
        
        for productIds, imageBatchIndices, XData in bsonIterator.IterGroupedBatches():
            print("Predict %d/%d (%.2f %%) (batch %d)" % (imagesProcessed, \
                bsonIterator.imagesMetaDf.shape[0],
                100 * imagesProcessed / bsonIterator.imagesMetaDf.shape[0],
                XData.shape[0]))
            
            # Predict
            activations = GetActivations(XData)
            
            # Combine multi-image predictions
            for productId, ids in zip(productIds, imageBatchIndices):
                if evaluate:
                    trueCategory = bsonIterator.productsMetaDf.loc[productId, "categoryId"]
                        
                for predictMethodName, predictFunc in predictMethods.items():
                    productActivations = predictFunc(activations[ids, :])
                    predictedClass = np.argmax(productActivations, axis = 0)
                    predictedCategory = self._mapClassToCategory[predictedClass]
                    
                    if evaluate:
                        if predictedCategory == trueCategory:
                            correctPredictions[predictMethodName] += 1
                    
                    if predictMethodName == finalPredictMethod:    
                        # Add to res
                        res.append([productId, predictedCategory])
                        
                        # Save top categories to bson
                        if topK > 0:
                            bestClasses = productActivations.argsort()[-topK:][::-1]
                            bestClassProbs = productActivations[bestClasses]
                            bestCategories = np.vectorize(lambda x: self._mapClassToCategory[x])(bestClasses)
                            resTopK.append([productId] + list(bestCategories) + list(bestClassProbs))
                            
                totalPredictions += 1
                
            # Print evaluation
            if evaluate:
                for predictMethodName in predictMethods:
                    print("Accuracy (%s) %.3f" % (predictMethodName, correctPredictions[predictMethodName] / totalPredictions))
            
            # Increase counters
            imagesProcessed += XData.shape[0]
        print("Predict done.")
        
        # ResDf
        resDf = pd.DataFrame(res, columns = ["_id", "category_id"])
        resDf.set_index("_id", inplace = True)
        
        # resTopKDf
        if topK > 0:
            resTopKDf = pd.DataFrame(resTopK, columns = ["_id"] + \
                ["pred_%d" % (i) for i in range(topK)] + ["prob_%d" % (i) for i in range(topK)])
            resTopKDf.set_index("_id", inplace = True)
        else:
            resTopKDf = None
            
        return resDf, resTopKDf
    
    def ValidateModel(self):
        df, topKDf = self.Predict(self.valGenerator, evaluate = True, topK = 5)
        df.to_csv(self.validationFilename + ".gz", compression = "gzip")
        topKDf.to_csv(self.validationTopKFilename + ".gz", compression = "gzip")
        
    def PrepareSubmission(self):
        print("PrepareSubmission...")
        df, topKDf = self.Predict(self.testGenerator, evaluate = False, topK = 5)
        df.to_csv(self.submissionFilename + ".gz", compression = "gzip")
        topKDf.to_csv(self.submissionTopKFilename + ".gz", compression = "gzip")
        print("PrepareSubmission done.")
           
    def _ReadCategoryTree(self):
        # Read
        df = pd.read_csv(path.join(self.datasetDir, "category_names.csv"))
        df.columns = ["categoryId", "categoryLevel1", "categoryLevel2", "categoryLevel3"]
        df["classId"] = range(df.shape[0])
        
        # Save
        self._dfCategories = df
        
        # Build index (categoryId -> classId) and (classId -> catecoryId)
        self._mapCategoryToClass = dict(zip(df.categoryId, df.classId))
        self._mapClassToCategory = dict(zip(df.classId, df.categoryId))
        
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
        
    def _MakeTrainValSets(self, productsMetaDf, splitPercentage = 0.1, dropoutPercentage = 0.0, seed = 0):
        np.random.seed(seed)
        indicesByGroups = productsMetaDf.groupby("categoryId", sort = False).indices
        numImgsColumnNr = productsMetaDf.columns.get_loc("numImgs")
        valMinSize = 0 if splitPercentage <= 0.0 else 1
        
        resVal, resTrain = [], []
        for _, indices in indicesByGroups.items():
            # Randomly drop
            toKeep = round(len(indices) * (1.0 - dropoutPercentage))
            if toKeep == 0:
                continue
            elif toKeep < len(indices):
                indices = np.random.choice(indices, toKeep, replace = False)
            
            # Validation set
            validationSize = max(min(valMinSize, len(indices) - 1), round(len(indices) * splitPercentage))
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
    def testDatasetName(self):
        return self.params["testDatasetName"]

    @property
    def targetSize(self):
        return self.params["targetSize"]

    @property
    def interpolationSize(self):
        return self.params["interpolationSize"]

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
    
    @property
    def submissionFilename(self):
        return path.join(self.trainingDir, "submission.csv")
    
    @property
    def validationFilename(self):
        return path.join(self.trainingDir, "validation.csv")

    @property
    def submissionTopKFilename(self):
        return path.join(self.trainingDir, "submissionTopK.csv")
    
    @property
    def validationTopKFilename(self):
        return path.join(self.trainingDir, "validationTopK.csv")


if __name__ == "__main__":
    pass
