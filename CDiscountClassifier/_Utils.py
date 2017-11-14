import io
import bson
import struct
import numpy as np
import pandas as pd
from os import path
from keras.preprocessing.image import load_img, img_to_array

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
    df.to_csv(outFile, index = False)
    
def ExtractAndPreprocessImg(productDict, imgNr, targetSize, imageDataGenerator):
    # Load image
    imgBytes = productDict["imgs"][imgNr]["picture"]
    img = load_img(io.BytesIO(imgBytes), target_size = targetSize)
    
    # Transform and standardize
    x = img_to_array(img)
    x = imageDataGenerator.random_transform(x)
    x = imageDataGenerator.standardize(x)
    
    return x
    
if __name__ == "__main__":
    pass
        