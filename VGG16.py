import tensorflow as tf
from keras.applications.vgg16 import VGG16


loss = tf.keras.losses.MeanSquaredError()

model = VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=max,
    classes=1000,
    classifier_activation="softmax",
)

def loss_gg(y_true, y_pred):
    return(loss(model(y_true), model(y_pred)))
