from fnmatch import fnmatch
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import resnet50, xception

def _SetLayerTrainable(model, pattern, trainable):
    for layer in model.layers:
            if fnmatch(layer.name, pattern):
                layer.trainable = trainable

def MyXception(imageShape, nClasses, modeTrainable = "onlyTop"):
    modelBase = xception.Xception(include_top = False, input_shape = imageShape, \
                                  weights = "imagenet")
    modelBase.trainable = False
    modelBase.summary()
    
    # Add top
    x = modelBase.outputs[0]
    x = GlobalAveragePooling2D(name = "avg_pool")(x)
    x = Dense(nClasses, activation = "softmax", name = 'predictions')(x)
  
    model = Model(modelBase.inputs, x, name='xception')
    
    # Freeze model
    _SetLayerTrainable(model, "*", False)
    
    # Enable
    if modeTrainable == "onlyTop":
        _SetLayerTrainable(model, "predictions", True)
    if modeTrainable == "full":
        _SetLayerTrainable(model, "*", True)
    else:
        ValueError("Unknown modeTrainable %s" % (modeTrainable))
    
    return model

PREPROCESS_FUNCS = {"Xception": xception.preprocess_input}
MODELS = {"Xception": MyXception}

