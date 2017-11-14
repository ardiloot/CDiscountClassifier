import os
import bson
import math
import numpy as np
import pandas as pd
import cProfile, pstats
import keras

from os import path
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.applications import resnet50
from keras.applications import xception

from CDiscountClassifier._Utils import PrecalcDatasetMetadata, ExtractAndPreprocessImg, BSONIterator
from CDiscountClassifier._HelperFunctions import RepeatAndLabel

class CDiscountClassfier:
    
    def __init__(self, datasetDir, seed = None):
        self.datasetDir = datasetDir
        self._targetSize = (180, 180)
        self._batchSize = 32
        self._seed = seed
         
        self._trainImageDataGenerator = ImageDataGenerator(preprocessing_function = xception.preprocess_input)
        self._valImageDataGenerator = ImageDataGenerator(preprocessing_function = xception.preprocess_input)
         
        self._ReadCategoryTree()
                
    def TrainModel(self, datasetName):
        # Filenames
        bsonFile = path.join(self.datasetDir, "%s.bson" % (datasetName))
        productsMetaFile = path.join(self.datasetDir, "%s_metadata.csv" % (datasetName))
        
        # Read metadata
        print("Loading metadata...")
        if not path.isfile(productsMetaFile):
            PrecalcDatasetMetadata(datasetName, self.datasetDir)
        productsMetaDf = pd.read_csv(productsMetaFile, index_col = "productId")
        productsMetaDf["classId"] = productsMetaDf.categoryId.map(self._mapCategoryToClass)
        print("Metadata loaded.")
        
        print(productsMetaDf.head())
        
        # Split to train and val
        print("Making train/val splits...")
        trainMetaDf, valMetaDf = self._MakeTrainValSets(productsMetaDf, \
            splitPercentage = 0.2, dropoutPercentage = 0.99, seed = self._seed)
        print("Train", trainMetaDf.shape)
        print("Val", valMetaDf.shape)
        
        # Iterators
        trainGenerator = BSONIterator(bsonFile, productsMetaDf, trainMetaDf, self.nClasses, self._trainImageDataGenerator,
                 self._targetSize, withLabels = True, batchSize = self._batchSize)

        valGenerator = BSONIterator(bsonFile, productsMetaDf, valMetaDf, self.nClasses, self._trainImageDataGenerator,
                 self._targetSize, withLabels = True, batchSize = self._batchSize)

        # Model
        modelBase = xception.Xception(include_top = False, input_shape = self.imageShape, weights = "imagenet", pooling = "avg")
        modelBase.summary()
        modelBase.trainable = False
        
        model = Sequential()
        model.add(modelBase)
        model.add(Dense(self.nClasses, activation = "softmax", name = "predictions"))
        model.summary()
        
        model.compile(loss = keras.losses.categorical_crossentropy,
                      optimizer = keras.optimizers.SGD(lr = 0.045, momentum = 0.9, decay = 0.94))
        
        # Fit
        model.fit_generator(trainGenerator,
                    steps_per_epoch = trainMetaDf.shape[0] // self._batchSize,
                    epochs = 5,
                    validation_data = valGenerator,
                    validation_steps = max(1, valMetaDf.shape[0] // self._batchSize),
                    workers = 5)
        

        
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
        
    def _MakeTrainValSets(self, productsMetaDf, splitPercentage = 0.2, dropoutPercentage = 0.0, seed = None):
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
            validationSize = round(len(indices) * splitPercentage)
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
        
        
    def _LoadImages(self, bsonFile, metadataDf, imageDataGenerator, numClasses, withLabels = True):
        imageShape = self._targetSize + (3,)
        XData = np.zeros((metadataDf.shape[0],) + imageShape, dtype = K.floatx())
        if withLabels:
            yData = np.zeros((metadataDf.shape[0], numClasses), dtype = K.floatx())
        
        with open(bsonFile, "rb") as file:
            for i in range(metadataDf.shape[0]):
                offset = metadataDf.offset[i]
                length = metadataDf.length[i]
                imgNr = metadataDf.imgNr[i]
                
                # Seek and read  
                file.seek(offset)
                imgBytes = file.read(length)
                productDict = bson.BSON.decode(imgBytes)
            
                x = ExtractAndPreprocessImg(productDict, imgNr, self._targetSize, imageDataGenerator)
            
                # Save
                XData[i] = x
                if withLabels:
                    classId = metadataDf.classId[i]
                    yData[i, classId] = 1.0
        return XData, yData

    @property
    def nClasses(self):
        return self._dfCategories.shape[0]

    @property
    def imageShape(self):
        return tuple(self._targetSize + (3,))

if __name__ == "__main__":
    datasetDir = os.environ["CDISCOUNT_DATASET"]
    print("datasetDir", datasetDir)
    
    profile = cProfile.Profile()
    profile.enable()
    
    m = CDiscountClassfier(datasetDir, seed = 0)
    m.TrainModel("train")
        
        
    profile.disable()
    pstats.Stats(profile).sort_stats("cumtime").print_stats(50) 
    
