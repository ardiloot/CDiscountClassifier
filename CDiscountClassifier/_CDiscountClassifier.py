import io
import bson
import math
import pylab as plt
import numpy as np
import pandas as pd
from os import path
from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras import backend as K
import cProfile, pstats
import threading
from CDiscountClassifier._Utils import PrecalcDatasetMetadata, ExtractAndPreprocessImg
from CDiscountClassifier._HelperFunctions import RepeatAndLabel

class BSONIterator(Iterator):
    def __init__(self, bsonFile, metadataDf, numClasses, imageDataGenerator,
                 targetSize, withLabels = True, batchSize = 32, 
                 shuffle = False, seed = None):

        self.file = open(bsonFile, "rb")
        self.metadataDf = metadataDf
        self.withLabels = withLabels
        self.samples = metadataDf.shape[0]
        self.numClasses = numClasses
        self.imageDataGenerator = imageDataGenerator
        self.targetSize = tuple(targetSize)
        self.imageShape = self.targetSize + (3,)

        super().__init__(self.samples, batchSize, shuffle, seed)
        self.lock = threading.Lock()

    def _get_batches_of_transformed_samples(self, indexArray):
        XBatch = np.zeros((len(indexArray),) + self.imageShape, dtype = K.floatx())
        if self.withLabels:
            yBatch = np.zeros((len(indexArray), self.numClasses), dtype = K.floatx())

        for batchId, sampleId in enumerate(indexArray):
            # Lock and read sample
            with self.lock:
                offset = self.metadataDf.offset[sampleId]
                length = self.metadataDf.length[sampleId]
                self.file.seek(offset)
                productDictBytes = self.file.read(length)

            # Extract and preprocess image
            productDict = bson.BSON.decode(productDictBytes)
            imgNr = self.metadataDf.imgNr[sampleId]
            x = ExtractAndPreprocessImg(productDict, imgNr, self.targetSize, self.imageDataGenerator)

            # Save
            XBatch[batchId] = x
            if self.withLabels:
                classId = self.metadataDf.classId[sampleId]
                yBatch[batchId, classId] = 1

        if self.withLabels:
            return XBatch, yBatch
        else:
            return XBatch

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)


class CDiscountClassfier:
    
    def __init__(self, datasetDir):
        self.datasetDir = datasetDir
        self._targetSize = (180, 180)
        self._batchSize = 32
         
        self._trainImageDataGenerator = ImageDataGenerator()
        self._valImageDataGenerator = ImageDataGenerator()
         
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
        print("Metadata loaded.")
        
        # Split to train and val
        print("Making train/val splits...")
        profile = cProfile.Profile()
        profile.enable()
        
        trainDf, valDf = self._MakeTrainValSets(productsMetaDf, splitPercentage = 0.2, dropoutPercentage = 0.0)
        
        profile.disable()
        pstats.Stats(profile).sort_stats("cumtime").print_stats(30)   
        print("Train", trainDf.shape)
        print("Val", valDf.shape)
        
        # Iterator
        trainGenerator = BSONIterator(bsonFile, productsMetaDf, dd, self.nClasses, self._trainImageDataGenerator,
                 self._targetSize, withLabels = True, batchSize = self._batchSize)

        profile = cProfile.Profile()
        profile.enable()
        
        count = 0
        for i, (x, y) in enumerate(trainGenerator):
            if i >= 1000:
                break
            count += x.shape[0]
                #print(x.shape, y.shape)
        profile.disable()
        pstats.Stats(profile).sort_stats("cumtime").print_stats(30)             
        print(count)

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
        
    def _MakeTrainValSets(self, productsMetaDf, splitPercentage = 0.2, dropoutPercentage = 0.0):
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
                productDict = bson.BSON.decode(file.read(length))
            
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

if __name__ == "__main__":

    
    m = CDiscountClassfier(r"D:\Kaggle")
    m.TrainModel("train_example")
    
    plt.show()

