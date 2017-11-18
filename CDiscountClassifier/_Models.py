from fnmatch import fnmatch
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import resnet50, xception

def _SetLayerTrainable(model, pattern, trainable):
    for layer in model.layers:
            if fnmatch(layer.name, pattern):
                layer.trainable = trainable

def MyXception(imageShape, nClasses, trainable = "onlyTop", trainableFromBlock = None):
    modelBase = xception.Xception(include_top = False, input_shape = imageShape, \
                                  weights = "imagenet")
    
    # Add top
    x = modelBase.outputs[0]
    x = GlobalAveragePooling2D(name = "avg_pool")(x)
    x = Dense(nClasses, activation = "softmax", name = 'predictions')(x)
  
    model = Model(modelBase.inputs, x, name='xception')
    
    # Freeze model
    _SetLayerTrainable(model, "*", False)
    
    # Enable
    if trainable == "onlyTop":
        _SetLayerTrainable(model, "predictions", True)
    elif trainable == "full":
        _SetLayerTrainable(model, "*", True)
    elif trainable == "blocks":
        for i in range(trainableFromBlock, 15):
            _SetLayerTrainable(model, "block%d_*" % (i), True)
        _SetLayerTrainable(model, "predictions", True)
    else:
        raise ValueError("Unknown trainable mode %s" % (trainable))
    
    return model

PREPROCESS_FUNCS = {"Xception": xception.preprocess_input}
MODELS = {"Xception": MyXception}

