from keras.models import Sequential
from keras.layers import Dense
from keras.applications import resnet50, xception

def MyXception(imageShape, nClasses, **kwargs):
    modelBase = xception.Xception(include_top = False, input_shape = imageShape, \
                                  weights = "imagenet", pooling = "avg", **kwargs)
    modelBase.trainable = False
    modelBase.summary()
    
    model = Sequential()
    model.add(modelBase)
    model.add(Dense(nClasses, activation = "softmax", name = "predictions"))
    model.summary()
    
    return model

PREPROCESS_FUNCS = {"Xception": xception.preprocess_input}
MODELS = {"Xception": MyXception}

