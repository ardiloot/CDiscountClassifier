import io
import bson
import struct
import threading
import numpy as np
import pandas as pd
from os import path
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array, Iterator


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
    
if __name__ == "__main__":
    pass
        