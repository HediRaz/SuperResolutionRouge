import tensorflow as tf
from tensorflow import keras



def create_model(B,dim):

    x_in = tf.keras.layers.Input(shape = dim)
    x = keras.layers.Conv2D(64, 9, padding='same')(x_in)
    x = keras.layers.PReLU()(x)
    x2 = x
    for i in range(B):
        x1 = keras.layers.Conv2D(64, 3, padding='same')(x2)
        x1 = keras.layers.BatchNormalization()(x1)
        x1 = keras.layers.PReLU()(x1)
        x1 = keras.layers.Conv2D(64, 3, padding = 'same')(x1)
        x1 = keras.layers.BatchNormalization()(x1)
        x1 = x1 + x2
        x2 = x1
    x1 = keras.layers.Conv2D(64, 3, padding = 'same')(x1)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = x1 + x
    x1 = keras.layers.Conv2D(256, 3, padding = 'same')(x1)

    x1 = keras.layers.PReLU()(x1)
    x1 = tf.nn.depth_to_space(x1, 2)
    x1 = keras.layers.Conv2D(256, 3, padding = 'same')(x1)
    x1 = tf.nn.depth_to_space(x1, 2)
    x1 = keras.layers.PReLU()(x1)
    x1 = keras.layers.Conv2D(3, 9, padding = 'same')(x1)
    x_out = x1

    return keras.Model(inputs=x_in , outputs=x_out)


def train_model(x_train, y_train,epochs, model, lossIn, batch_size):
    model.compile(optimizer = 'adam', loss = lossIn)
    return model.fit(x_train, y_train, epochs = epochs, batch_size=batch_size)
