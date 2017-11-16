import io
import bson
import struct
import threading
import keras

import numpy as np
import pandas as pd

from os import path
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array, Iterator

#===============================================================================
# Functions
#===============================================================================

def BuildDatasetMetadata(datasetName, datasetDir):
    # https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
    rows = []
    with open(path.join(datasetDir, "%s.bson" % (datasetName)), "rb") as f:
        offset = 0
        while True:
            # Break condition
            itemLengthBytes = f.read(4)
            if len(itemLengthBytes) == 0:
                break

            # Read length
            length = struct.unpack("<i", itemLengthBytes)[0]

            # Read data
            f.seek(offset)
            itemData = f.read(length)
            assert len(itemData) == length

            # Decode
            item = bson.BSON.decode(itemData)
            productId = item["_id"]
            numImgs = len(item["imgs"])

            # Add to rowsDict
            row = [productId, numImgs, offset, length] + ([item["category_id"]] if "category_id" in item else [-1])
            rows.append(row)

            # Update offset
            offset += length
            f.seek(offset)

    # Convert tict to DataFrame
    df = pd.DataFrame(rows, columns = ["productId", "numImgs", "offset", "length", "categoryId"])
    df.set_index("productId", inplace = True)
    df.sort_index(inplace = True)
    return df

def PrecalcDatasetMetadata(datasetName, datasetDir):
    df = BuildDatasetMetadata(datasetName, datasetDir)
    outFile = path.join(datasetDir, "%s_metadata.csv" % (datasetName))
    df.to_csv(outFile)
    
def ExtractAndPreprocessImg(productDict, imgNr, targetSize, imageDataGenerator):
    # Load image
    imgBytes = productDict["imgs"][imgNr]["picture"]
    img = load_img(io.BytesIO(imgBytes), target_size = targetSize)
    
    # Transform and standardize
    x = img_to_array(img)
    x = imageDataGenerator.random_transform(x)
    x = imageDataGenerator.standardize(x)
    
    return x
    
#===============================================================================
# BSONIterator
#===============================================================================

class BSONIterator(Iterator):
    def __init__(self, bsonFile, productsMetaDf, imagesMetaDf, numClasses, imageDataGenerator,
                 targetSize, withLabels = True, batchSize = 32, 
                 shuffle = False, seed = None):

        self.file = open(bsonFile, "rb")
        self.productsMetaDf = productsMetaDf
        self.withLabels = withLabels
        self.imagesMetaDf = imagesMetaDf
        self.samples = imagesMetaDf.shape[0]
        self.numClasses = numClasses
        self.imageDataGenerator = imageDataGenerator
        self.targetSize = tuple(targetSize)
        self.imageShape = self.targetSize + (3,)

        super().__init__(self.samples, batchSize, shuffle, seed)
        self.fileLock = threading.Lock()

    def _get_batches_of_transformed_samples(self, indexArray):
        XBatch = np.zeros((len(indexArray),) + self.imageShape, dtype = K.floatx())
        if self.withLabels:
            yBatch = np.zeros((len(indexArray), self.numClasses), dtype = K.floatx())

        for batchId, sampleId in enumerate(indexArray):
            # Lock and read sample
            with self.fileLock:
                # Image data
                imageData = self.imagesMetaDf.iloc[sampleId]
                productId = imageData["productId"]
                imgNr = imageData["imgNr"]
                
                # Product data
                productData = self.productsMetaDf.loc[productId]
                offset = productData["offset"]
                length = productData["length"]
                classId = productData["classId"]

                # Read file
                self.file.seek(offset)
                productDictBytes = self.file.read(length)

            # Extract and preprocess image
            productDict = bson.BSON.decode(productDictBytes)
            x = ExtractAndPreprocessImg(productDict, imgNr, self.targetSize, self.imageDataGenerator)

            # Save
            XBatch[batchId] = x
            if self.withLabels:
                yBatch[batchId, classId] = 1

        if self.withLabels:
            return XBatch, yBatch
        else:
            return XBatch

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)
 
 #==============================================================================
 # TrainTimeStatsCallback
 #==============================================================================
 
class TrainTimeStatsCallback(keras.callbacks.Callback):

    def __init__(self, filename, statsPerEpoch = 100, \
                 toSave = ["loss", "acc", "val_loss", "val_acc"]):
        self.filename = filename
        self.statsPerEpoch = statsPerEpoch
        self.toSave = toSave
        self.curEpoch = None
        self.curBatch = None
        self.file = None
        super().__init__()

    def on_train_begin(self, logs = None):
        self.verbose = self.params["verbose"]
        self.epochs = self.params["epochs"]
        self.statsBatchNrDelta = max(1, self.params["steps"] // self.statsPerEpoch)

        # Open file
        printHeader = not path.isfile(self.filename)
        self.file = open(self.filename, "a")
        if printHeader:
            self.file.write("%s\n" % ("\t".join(["Epoch"] + self.toSave)))

    def on_train_end(self, logs = None):
        if self.file is not None:
            self.file.close()

    def on_epoch_begin(self, epoch, logs = None):
        self.curEpoch = epoch

    def on_epoch_end(self, epoch, logs = None):
        self.SaveStats(logs)

    def on_batch_begin(self, batch, logs = None):
        self.curBatch = batch

    def on_batch_end(self, batch, logs = None):
        if batch % self.statsBatchNrDelta != 0 or batch + 1 == self.params["steps"]:
            return
        self.SaveStats(logs)
        
    def SaveStats(self, logs):
        epochFloat = self.curEpoch + (self.curBatch + 1) / self.params["steps"]
        logs = logs or {}
        
        row = [epochFloat]
        for metricName in self.toSave:
            if metricName in logs:
                row.append(logs[metricName])
            else:
                row.append("")
        self.file.write("%s\n" % ("\t".join(map(str, row))))
        self.file.flush()
        
if __name__ == "__main__":
    pass
        