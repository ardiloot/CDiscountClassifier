import os
import yaml
import keras
import numpy as np
import pandas as pd
import keras.backend as K

from os import path
from datetime import datetime
from CDiscountClassifier import _Models
from CDiscountClassifier._Utils import PrecalcDatasetMetadata, BSONIterator, \
    TrainTimeStatsCallback, SetEpochParams, MultiGPUModelCheckpoint, \
    CropImageDataGenerator, SGDAccum, AdamAccum
from CDiscountClassifier._HelperFunctions import RepeatAndLabel  # @UnresolvedImport


#===============================================================================
# CDiscountClassfier
#===============================================================================

class CDiscountClassfier:
    """Main class used for the Kaggle competition "Cdiscount’s Image
    Classification Challenge". This class is really general and configurable
    through parameters.
    
    Parameters
    ----------
    friendlyName : string
        A freiendly name for a model. If not specified, model name will be used.
    datasetDir : string
        Path to the directory containing training and testing BSON files. If not
        specified, environment variable CDISCOUNT_DATASET will be used instead.
    resultsDir : string
        Path to the directory for storing results files. Default is "../../results".
    trainDatasetName : string
        Name of the training dataset. In this case, train and train_example are
        possible. Default is train (full dataset)
    interpolationSize : tuple of ints
        After loading the image, the image will be interpolated to specified
        size. Default is (180, 180)
    interpolation : string
        Name of the interpolation method. Default is bicubic.
    targetSize : tuple of ints
        The final size of the image to be inputed to the neural network. If
        `interpolationSize` is not equal to `targetSize`, the image will be
        cropped either randomly or from center. Default is (180, 180).
    batchSize : int
        The size of the batch. Default is 64.
    epochs : int
        The number of epochs to train the network. Default is 5.
    workers : int
        Number of workers for loading image data to feed GPU. Default is 5.
    nTtaAugmentation : int
        Number of augmented images to use in case of testing. Default is 1 (no
        augmentation)
    trainSeed : int
        Seed for random number generation. Will be set before training. Default
        1000003.
    valImagesPerEpoch : int or None
        Number of validation images used for artificial epochs. If None, all
        available validation data is used (default).
    trainImagesPerEpoch : int or None
        Number of train images used for artificial epochs. If None, all
        available training data is used (default).
    trainAugmentation : dict
        The parameters for augmentation of training data. Accepts all parameters
        as `ImageDataGenerator` and additional params defined by
        `CropImageDataGenerator`.
    trainAugmentation : dict
        The parameters for augmentation of training data. Accepts all parameters
        as `ImageDataGenerator` and additional params defined by
        `CropImageDataGenerator`.    
    valAugmentation : dict
        The parameters for augmentation of validation data. Accepts all parameters
        as `ImageDataGenerator` and additional params defined by
        `CropImageDataGenerator`.
    testAugmentation : dict
        The parameters for augmentation of test data. Accepts all parameters
        as `ImageDataGenerator` and additional params defined by
        `CropImageDataGenerator`.    
    predictMethod : string
        Name of the main predict method used for combining the predictions of
        multiple images. Possible selections are "meanActivations", "median",
        "pwr0.2", "pwr0.1", "pwr0.05", "max", "firstImage".
    testDropout : float
        Fraction of test data to drop. Used for fast testing. Default is 0.0.
    valTrainSplit : dict
        Parameters used for making train and validation split. For parameter
        definitions see the description of the method `_MakeTrainValSets`.
    model : dict
        Parameter dictionary for model. For possible parameters see _Models module.
    optimizer : dict
        Contains to "name" (SGD, Adam, AdamAccum) of the optimizer and parameters
        for it ("kwargs").
    epochSpecificParams : dict
        Dictionary for epoch specific parameters. Makes possible to tune learning
        rate and model trainability turing the optimization.
        
    Attributes
    ----------
    params : dict
        Dictionary containing all classifier parameters.
    datasetDir : string
        Path to dataset directry.
    trainDatasetName : string
        Name of the training dataset to use.
    targetSize : tuple of ints
        Size of the images used by neural networks.
    interpolationSize : tuple of ints
        Size for rezizing the images.
    batchSize : int
        The size of the batch.
    nTtaAugmentation : int
        Number of test time augmentation.
    nClasses : int
        Number of different classification classes.
    imageShape : tuple of ints
        Raw image data shape.
    resultsDir : string
        Path to results directory.
    trainingDir : string
        Path to the results folder of the current model run.
    modelFilename : string
        Path for the `ModelCheckpoint` callback.
    statsFilename : string
        Path to the additional accuracy log CSV file.
    logFilename : string
        Path to the log file.
    submissionFilename : string
        Path to the submission CSV file.
    validationFilename : string
        Path to the validation CSV file.
    submissionTopKFilename : string
        Path to the top-K activations file of submission.
    validationTopKFilename : string
        Path to the top-K activations file of validation.
        
    """
    
    def __init__(self, **kwargs):
        # Default params
        self.params = {
            "friendlyName": None,
            "datasetDir": None,
            "resultsDir": "../../results",
            "trainDatasetName": "train",
            "testDatasetName": "test",
            "interpolationSize": (180, 180),
            "interpolation": "nearest",
            "targetSize": (180, 180),
            "batchSize": 64,
            "epochs": 5,
            "workers": 5,
            "nTtaAugmentation": 1,
            "trainSeed": 1000003,
            "valImagesPerEpoch": None,
            "trainImagesPerEpoch": None,
            "trainAugmentation": {},
            "valAugmentation": {},
            "testAugmentation": {},
            "predictMethod": "meanActivations",
            "testDropout": 0.0,
            "valTrainSplit": {
                "splitPercentage": 0.2,
                "dropoutPercentage": 0.0,
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
            "epochSpecificParams": {}
        }
                  
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
        """This method generates a unique training name. If friendlyName is
        specified, then it will be concatenaded with current time. If
        freindlyName is None, then model name will be used instead. Usually
        called internally by method `TrainModel`, but in some cases, is usful
        to generate training name beforehand.
        
        Returns
        -------
        string
            Unique training name.
        
        """
        
        dateStr = datetime.now().strftime("%Y%m%d-%H%M%S")
        friendlyName = self.params["model"]["name"] if self.params["friendlyName"] is None else self.params["friendlyName"] 
        self.trainingName = "%s_%s" % (dateStr, friendlyName)
        return self.trainingName 
                
    def InitTrainingData(self):
        """Calling this method prepares training data for use. Must be called
        before calling method `TrainModel`. 
        
        """
        
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
            withLabels = True, batchSize = self.batchSize, shuffle = True, interpolation = self.params["interpolation"])

        self.valGenerator = BSONIterator(bsonFile, self.trainProductsMetaDf, self.valMetaDf, \
            self.nClasses, valImageDataGenerator, interpolationSize = self.interpolationSize, \
            withLabels = True, batchSize = self.batchSize, shuffle = True, lock = self.trainGenerator.lock, \
            interpolation = self.params["interpolation"])
        
        print("Init iterators done.")
   
    def InitTestData(self):
        """Calling this method prepares test data for use. Must be called
        before calling method `PrepareSubmission`. 
        
        """
        
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
            withLabels = False, batchSize = self.batchSize, shuffle = False, interpolation = self.params["interpolation"])
        print("Init iterators done.")
        print("Init test data done.")
                
    def TrainModel(self, updateTrainingName = True, newTrainingName = None):
        """Trains the model specifed by parameters.
        
        Parameters
        ----------
        updateTrainingName : bool
            If True (default), a new training name (and training results folder)
            will be generated. If False, old trainingName will be used.
        newTrainingName : string
            Has only effect if `updateTrainingName` is True. If is none, a
            `GenerateTrainingName` method will be used for a new name. If not
            None, the value specifed by `newTrainingName` will be used as a new
            name.
        
        """
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
        elif params["optimizer"]["name"] == "AdamAccum":
            optimizer = AdamAccum(**params["optimizer"]["kwargs"])
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
                workers = self.params["workers"],
                initial_epoch = curEpoch)
            print("Fit generator done.")
            
        self.model = model
        print("Model fit done.")
                
    def Predict(self, bsonIterator, evaluate = False, topK = 0, nAugmentation = 1):
        """Main method for evaluation and predictions by a trained model. This
        method is also used by `ValidateModel` and `PrepareSubmission`.
        
        Parameters
        ----------
        bsonIterator : `BSONIterator`
            A incatance of `BSONIterator` with a data to be used for predictions.
        evaluate : bool
            If True, the predictions will be also evaluated.
        topK : int
            Number of top categories to save.
        nAugmentation : int
            Number of augmented images to use for test time augmentation.
            
        Returns
        -------
        resDf : `DataFrame`
            DataFrame containing productIds and predicted categories.
        (resSaveProductIds, resSaveCategories, resSaveActivations) : tuple of ndarray
            ProductIds, categories and activations for topK classes for each
            product.
        
        """
        
        print("Predict")
        
        predictMethods = {
            "meanActivations": lambda x: np.mean(x, axis = 0),
            "productActivations": lambda x: np.prod(x, axis = 0),
            "rmsActivations": lambda x: np.mean(x ** 2.0, axis = 0),
            "median": lambda x: np.median(x, axis = 0),
            "pwr0.2": lambda x: np.mean(x ** 0.2, axis = 0),
            "pwr0.1": lambda x: np.mean(x ** 0.1, axis = 0),
            "pwr0.05": lambda x: np.mean(x ** 0.05, axis = 0),
            "max": lambda x: np.max(x, axis = 0),
            "firstImage": lambda x: x[0, :],
            }
        
        finalPredictMethod = self.params["predictMethod"]
        if finalPredictMethod not in predictMethods:
            raise ValueError("Unknown predictMethod" , finalPredictMethod)
        
        GetActivations = K.function([K.learning_phase()] + self.model.inputs, self.model.outputs)
       
        res = dict((metricName, []) for metricName in predictMethods)
        correctPredictions = dict((k, 0) for k in predictMethods)
        imagesProcessed = 0
        totalPredictions = 0
        resSaveCount = 0
        resSaveActivations = np.zeros((bsonIterator.productCount, topK), dtype = np.float32)
        resSaveCategories = np.zeros((bsonIterator.productCount, topK), dtype = np.int64)
        resSaveProductIds = np.zeros((bsonIterator.productCount,), dtype = np.int64)
        
        for productIds, imageBatchIndices, XData in bsonIterator.IterGroupedBatches(workers = self.params["workers"], nAugmentation = nAugmentation):
            print("Predict %d/%d (%.2f %%) (batch %d)" % (imagesProcessed, \
                bsonIterator.imagesMetaDf.shape[0] * nAugmentation,
                100 * imagesProcessed / bsonIterator.imagesMetaDf.shape[0] / nAugmentation,
                XData.shape[0]))
            
            # Predict
            activations = GetActivations([0, XData])[0]
            
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
                    
                    # Add to res
                    res[predictMethodName].append([productId, predictedCategory])
                    
                    if predictMethodName == finalPredictMethod:    
                        if topK > 0:                
                            bestClasses = productActivations.argsort()[-topK:][::-1]
                            bestActivations = productActivations[bestClasses]
                            bestCategories = np.vectorize(lambda x: self._mapClassToCategory[x])(bestClasses)
                            
                            resSaveActivations[resSaveCount, :] = bestActivations
                            resSaveCategories[resSaveCount, :] = bestCategories
                            resSaveProductIds[resSaveCount] = productId
                            resSaveCount += 1
                       
                totalPredictions += 1
                
            # Print evaluation
            if evaluate:
                for predictMethodName in predictMethods:
                    print("Accuracy (%s) %.3f" % (predictMethodName, correctPredictions[predictMethodName] / totalPredictions))
            
            # Increase counters
            imagesProcessed += XData.shape[0]

        print("Predict done.")
        
        # ResDf
        resDf = {}
        for metricName, r in res.items():
            df = pd.DataFrame(r, columns = ["_id", "category_id"])
            df.set_index("_id", inplace = True)
            resDf[metricName] = df
                
        return resDf, (resSaveProductIds, resSaveCategories, resSaveActivations)
    
    def ValidateModel(self):
        """Validates model against full set of validation data and saves results
        to `trainingDir`.
        
        """
        
        dfs, (resSaveProductIds, resSaveCategories, resSaveActivations) = \
            self.Predict(self.valGenerator, evaluate = True, topK = 100, \
            nAugmentation = self.nTtaAugmentation)
         
        for metricName, df in dfs.items():   
            df.to_csv(self.validationFilename + metricName + ".csv.gz", compression = "gzip")

        np.save(self.validationTopKFilename + "_products", resSaveProductIds)
        np.save(self.validationTopKFilename + "_categories", resSaveCategories)
        np.save(self.validationTopKFilename + "_activations", resSaveActivations)
        
    def PrepareSubmission(self):
        """Makes predictions for every test product and saves the results to
        `trainingDir`.
        
        """
        
        print("PrepareSubmission...")
        dfs, (resSaveProductIds, resSaveCategories, resSaveActivations) = \
            self.Predict(self.testGenerator, evaluate = False, topK = 100,
            nAugmentation = self.nTtaAugmentation)
        for metricName, df in dfs.items():
            df.to_csv(self.submissionFilename + metricName + ".csv.gz", compression = "gzip")
        
        np.save(self.submissionTopKFilename + "_products", resSaveProductIds)
        np.save(self.submissionTopKFilename + "_categories", resSaveCategories)
        np.save(self.submissionTopKFilename + "_activations", resSaveActivations)
        
        print("PrepareSubmission done.")
           
    # Heper functions
    # --------------------------------------------------------------------------
           
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
        """Splits dataset to train and validation set.
        
        Parameters
        ----------
        productsMetaDf : `DataFrame`
            Products metadata DataFrame.
        splitPercentage : float
            Validation split fraction.
        dropoutPercentage : float
            Dropout fraction.
        seed : int
            Specified seed for random number generator.
        
        
        """
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
    def nTtaAugmentation(self):
        return self.params["nTtaAugmentation"]
    
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
        return path.join(self.trainingDir, "submission_")
    
    @property
    def validationFilename(self):
        return path.join(self.trainingDir, "validation_")

    @property
    def submissionTopKFilename(self):
        return path.join(self.trainingDir, "submissionTopK")
    
    @property
    def validationTopKFilename(self):
        return path.join(self.trainingDir, "validationTopK")


if __name__ == "__main__":
    pass
