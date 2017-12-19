"""This module contains utilities for `CDiscountClassifier` class.

"""

import io
import bson
import struct
import threading
import keras

import numpy as np
import pandas as pd
import tensorflow as tf

from os import path
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils.data_utils import GeneratorEnqueuer
from keras.legacy import interfaces
from collections import deque, defaultdict

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
    
def ExtractAndPreprocessImg(productDict, imgNr, interpolationSize, imageDataGenerator, interpolation = "nearest"):
    # Load image
    imgBytes = productDict["imgs"][imgNr]["picture"]
    if np.random.random() < imageDataGenerator.cropProbability:
        imageSize = interpolationSize
        img = load_img(io.BytesIO(imgBytes), target_size = imageSize, interpolation = interpolation)
        x = img_to_array(img)
        x = imageDataGenerator.Crop(x)
    else:
        imageSize = imageDataGenerator.targetSize
        img = load_img(io.BytesIO(imgBytes), target_size = imageSize, interpolation = interpolation)
        x = img_to_array(img)
        
    # Transform and standardize
    x = imageDataGenerator.random_transform(x)
    x = imageDataGenerator.standardize(x)
    
    #import pylab as plt
    #plt.imshow(x.astype(np.uint8))
    #plt.show()
    
    return x
 
def SetEpochParams(model, curEpoch, epochSpecificParams):
    if curEpoch not in epochSpecificParams:
        return
    
    # Init
    params = epochSpecificParams[curEpoch]
    print("Update epoch specific params:", params)
    oldLr = keras.backend.get_value(model.optimizer.lr)
    
    if hasattr(model, "modelCPU"):
        realModel = model.modelCPU
    else:
        realModel = model
    
    # Update
    for k, v in params.items():
        if k == "lrDecayCoef":
            curLr = keras.backend.get_value(model.optimizer.lr)
            keras.backend.set_value(model.optimizer.lr, v * curLr)
        elif k == "trainable":
            realModel.SetTrainable(v)
        else:
            raise ValueError("Unknown param %s" % (k))
        
    # Print confirmations        
    if "trainable" in params:
        print("Need to recompile")
        oldOptimizerClass = model.optimizer.__class__
        oldOptimizerParams = model.optimizer.get_config()
        newOptimizer = oldOptimizerClass(**oldOptimizerParams)
        model.compile(metrics = ["accuracy"],
            loss = "categorical_crossentropy",
            optimizer = newOptimizer)
        
        model.summary()
        print("LR", oldLr, "->", keras.backend.get_value(model.optimizer.lr))
        
    if "lrDecayCoef" in params:
        print("LR", oldLr, "->", keras.backend.get_value(model.optimizer.lr))
    
#===============================================================================
# CropImageDataGenerator
#===============================================================================

class CropImageDataGenerator(ImageDataGenerator):
    
    def __init__(self, *args, targetSize = None, cropProbability = 0.0, cropMode = "center", **kwargs):
        super().__init__(*args, **kwargs)
        self.targetSize = tuple(targetSize)
        self.cropMode = cropMode
        self.cropProbability = cropProbability
        
    def Crop(self, x, mode = None):
        if mode is None:
            mode = self.cropMode
        
        # Crop
        delta = np.array(x.shape)[:2] - np.array(self.targetSize)    
        if mode == "center":
            origin = (delta / 2).astype(int)
        elif mode == "random":
            origin = np.array([np.random.random_integers(0, delta[0]),\
                               np.random.random_integers(0, delta[1])], dtype = int)
        else:
            raise ValueError("UnknownCropMode", mode)
    
        res = self._DoCrop(x, origin)
        #print("crop", mode, x.shape, res.shape, origin)
        return res
        
    def _DoCrop(self, x, origin):
        if (self.targetSize[0] + origin[0]) > x.shape[0]:
            raise ValueError("Invalid crop height", x.shape, origin, self.targetSize)  
        
        if (self.targetSize[1] + origin[1]) > x.shape[1]:
            raise ValueError("Invalid crop width", x.shape, origin, self.targetSize)
        
        if min(origin) < 0:
            raise ValueError("Invalid crop origin", x.shape, origin, self.targetSize)
        
        res = x[origin[0]:(origin[0] + self.targetSize[0]),\
                origin[1]:(origin[1] + self.targetSize[1]),\
                :]
        return res
        
    @property
    def imageShape(self):
        return self.targetSize + (3,)
    
#===============================================================================
# BSONIterator
#===============================================================================

class BSONIterator:
    def __init__(self, bsonFile, productsMetaDf, imagesMetaDf, numClasses, imageDataGenerator,
                 interpolationSize = None, withLabels = True, batchSize = None, 
                 shuffle = True, seed = None, interpolation = "nearest", lock = None):

        self.bsonFile = bsonFile
        self.productsMetaDf = productsMetaDf
        self.imagesMetaDf = imagesMetaDf
        self.numClasses = numClasses
        self.imageDataGenerator = imageDataGenerator
        self.interpolationSize = tuple(interpolationSize)
        
        self.withLabels = withLabels
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.seed = seed
        self.interpolation = interpolation
        self.lock = lock
        
        if self.lock is None:
            self.lock = threading.RLock()

        self.file = open(self.bsonFile, "rb")
        self.samples = self.imagesMetaDf.shape[0]
        self.productCount = self.imagesMetaDf.productId.nunique()
        self.nextIndex = 0
        self.totalBatchesSeen = 0
        self._UpdateIndexArray()

    def _UpdateIndexArray(self):
        with self.lock:
            # Set seed
            if self.seed is not None:
                np.random.seed(self.seed + self.totalBatchesSeen)
        
            # Index array
            if self.shuffle:
                self.indexArray = np.random.permutation(self.samples)
            else:
                self.indexArray = np.array(range(self.samples))
            self.nextIndex = 0

    def __next__(self):
        with self.lock:
            if self.nextIndex + self.batchSize > self.samples:
                # Need to resuffle
                self._UpdateIndexArray()
            
            indices = self.indexArray[self.nextIndex:self.nextIndex + self.batchSize].copy()
            self.nextIndex += self.batchSize
            self.totalBatchesSeen += 1
            
        res = self._GetBatchesOfTransformedSamples(indices)
        return res
 
    def IterBatches(self):
        with self.lock:
            for index in range(0, self.samples, self.batchSize):
                indices = np.array(range(index, min(self.samples, index + self.batchSize)))
                r = self._GetBatchesOfTransformedSamples(indices)
                yield r
    
    def IterGroupedBatches(self, workers = 5, maxQueueSize = 10, nAugmentation = 1, withLabels = False):
        
        class BatchesIterator():
            
            def __init__(self, bsonIterator, batches, withLabels = False):
                self.bsonIterator = bsonIterator
                self.batches = batches
                self.withLabels = withLabels
                
                self.lock = threading.RLock()
                self.index = 0
                
            def __next__(self):
                with self.lock:
                    if self.index >= len(self.batches):
                        raise StopIteration()
                    
                    productIds, imageMetaIndices, imageBatchIndices = self.batches[self.index]
                    self.index += 1
                data = self.bsonIterator._GetBatchesOfTransformedSamples(\
                    imageMetaIndices, withLabels = self.withLabels)
                return productIds, imageBatchIndices, data
        
        with self.lock:
            # Prepare batch indices
            batches = []
            groupedByProducts = list(self.imagesMetaDf.groupby("productId").indices.items())
            groupIndex = 0
            while groupIndex < len(groupedByProducts): 
                # Make batch
                productIds = []
                imageMetaIndices = []
                imageBatchIndices = []
                batchLen = 0
                while groupIndex < len(groupedByProducts) and batchLen < self.batchSize:
                    productId, ids = groupedByProducts[groupIndex]
                    if batchLen + nAugmentation * len(ids) > self.batchSize:
                        break
                    
                    productIds.append(productId)
                    imageMetaIndices.append(np.repeat(ids, nAugmentation))
                    imageBatchIndices.append(range(batchLen, batchLen + nAugmentation * len(ids)))
                    batchLen += nAugmentation * len(ids)
                    groupIndex += 1
                    
                batches.append((productIds, np.concatenate(imageMetaIndices), imageBatchIndices))
            
        # Threaded data aquisition
        generator = BatchesIterator(self, batches, withLabels = withLabels)
        try:
            # Prepare queue
            enqueuer = GeneratorEnqueuer(generator)
            enqueuer.start(workers = workers, max_queue_size = maxQueueSize)
            outputGenerator = enqueuer.get()
            
            for r in outputGenerator:
                yield r
            
        except StopIteration:
            pass
        finally:
            if enqueuer is not None:
                enqueuer.stop()
            
    def _GetBatchesOfTransformedSamples(self, indexArray, withLabels = None):
        if withLabels is None:
            withLabels = self.withLabels
            
        XBatch = np.zeros((len(indexArray),) + self.imageDataGenerator.imageShape, dtype = K.floatx())
        if withLabels:
            yBatch = np.zeros((len(indexArray), self.numClasses), dtype = K.floatx())

        for batchId, sampleId in enumerate(indexArray):
            # read image
            x, classId = self._LoadImg(self.imagesMetaDf, sampleId)

            # Save
            XBatch[batchId] = x
            if withLabels:
                yBatch[batchId, classId] = 1

        if withLabels:
            return XBatch, yBatch
        else:
            return XBatch

    def _LoadImg(self, df, sampleId):
        with self.lock:
            # Image data
            imageData = df.iloc[sampleId]
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
        x = ExtractAndPreprocessImg(productDict, imgNr, self.interpolationSize, \
                                    self.imageDataGenerator, interpolation = self.interpolation)
        
        return x, classId

#==============================================================================
# TrainTimeStatsCallback
#==============================================================================
 
class TrainTimeStatsCallback(keras.callbacks.Callback):

    def __init__(self, filename, statsPerEpoch = 100, \
                 toSave = ["loss", "acc", "loss_avg", "acc_avg", "val_loss", "val_acc"]):
        self.filename = filename
        self.statsPerEpoch = statsPerEpoch
        self.toSave = toSave
        self.curEpoch = None
        self.curBatch = None
        self.file = None
        self.runningMeanDeque = None
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
        self.runningMeanDeque = defaultdict(lambda: deque(maxlen = self.statsBatchNrDelta))
        
    def on_epoch_end(self, epoch, logs = None):
        self.SaveStats(logs)

    def on_batch_begin(self, batch, logs = None):
        self.params["steps"]
        self.curBatch = batch

    def on_batch_end(self, batch, logs = None):
        for metricName in self.toSave:
            if metricName in logs:
                self.runningMeanDeque["%s_avg" % (metricName)].append(logs[metricName])
            
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
            elif metricName in self.runningMeanDeque:
                meanValue = np.mean(self.runningMeanDeque[metricName])
                row.append(meanValue)
                
            else:
                row.append("")
        self.file.write("%s\n" % ("\t".join(map(str, row))))
        self.file.flush()
     
#===============================================================================
# MultiGPUModelCheckpoint
#===============================================================================

class MultiGPUModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def set_model(self, model):
        if hasattr(model, "modelCPU"):
            print("MultiGPUModelCheckpoint:", "Using multiple GPUs")
            self.model = model.modelCPU
        else:
            print("MultiGPUModelCheckpoint:", "Using single/no GPUs")
            self.model = model
        
#===============================================================================
# SGDAccum
#===============================================================================

class SGDAccum(keras.optimizers.Optimizer):
    # https://github.com/fchollet/keras/issues/3556
    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, accum_iters=1, **kwargs):
        super(SGDAccum, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.accum_iters = K.variable(accum_iters, name='accum_iters')
        self.initial_decay = decay
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        accum_switch = K.equal(self.iterations % self.accum_iters, 0)
        accum_switch = K.cast(accum_switch, dtype='float32')

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        temp_grads = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, cg, m, tg in zip(params, grads, moments, temp_grads):
            g = cg + tg
            v = self.momentum * m - (lr * g / self.accum_iters)  # velocity
            self.updates.append(K.update(m, (1 - accum_switch) * m + accum_switch * v))
            self.updates.append(K.update(tg, (1 - accum_switch) * g))

            if self.nesterov:
                new_p = p + self.momentum * v - (lr * g / self.accum_iters)
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - accum_switch) * p + accum_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'accum_iters': int(K.get_value(self.accum_iters))}
        base_config = super(SGDAccum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#===============================================================================
# AdamAccum
#===============================================================================

class AdamAccum(keras.optimizers.Optimizer):
    def __init__(self, lr = 0.001, beta_1 = 0.9, beta_2 = 0.999,
                 epsilon = 1e-8, decay = 0., accum_iters = 1, **kwargs):
        super().__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype = 'int64', name = 'iterations')
            self.lr = K.variable(lr, name = 'lr')
            self.beta_1 = K.variable(beta_1, name = 'beta_1')
            self.beta_2 = K.variable(beta_2, name = 'beta_2')
            self.decay = K.variable(decay, name = 'decay')
            self.accum_iters = K.variable(accum_iters, dtype = 'int64', name='accum_iters')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        new_iter_op = tf.assign_add(self.iterations, 1)
        self.updates = []
    
        lr = self.lr
        with tf.control_dependencies([new_iter_op]):
            if self.initial_decay > 0:
                lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

            accum_switch = K.cast(K.equal(self.iterations % self.accum_iters, 0), dtype = K.floatx())
            t = K.cast(self.iterations // self.accum_iters, K.floatx()) + 1   
        accum_iters = K.cast(self.accum_iters, dtype = K.floatx())
              
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / 
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype = K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype = K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype = K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, gp, m, v, ga in zip(params, grads, ms, vs, gs):
            g = (ga + gp) / accum_iters
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - accum_switch) * m + accum_switch * m_t))
            self.updates.append(K.update(v, (1 - accum_switch) * v + accum_switch * v_t))
            self.updates.append(K.update(ga, (1 - accum_switch) * (ga + gp)))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
                
            self.updates.append(K.update(p, (1 - accum_switch) * p + accum_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'accum_iters': np.int64(K.get_value(self.accum_iters))}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
if __name__ == "__main__":
    pass
        
