import tensorflow as tf
from tensorflow import keras


def create_model_D(dim):

    x_in = tf.keras.layers.Input(shape = dim)

    x = keras.layers.Conv2D(64, 3, padding='same')(x_in)
    x = keras.layers.LeakyReLU()(x)
    #64
    x = keras.layers.Conv2D(64, 3, 2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    #128
    x = keras.layers.Conv2D(128, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(128, 3,2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    #256
    x = keras.layers.Conv2D(256, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(256, 3, 2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    #512
    x = keras.layers.Conv2D(512, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(512, 3, 2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1024)(x)
    x = keras.layers.LeakyReLU()(x)
    x_out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=x_in , outputs=x_out)
    

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

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def lossg(x):
    fake_loss = cross_entropy(tf.ones_like(x), x-0.0001)
    return fake_loss
def lossd(x, y):
    real_loss = cross_entropy(tf.ones_like(y), y-0.0001)
    fake_loss = cross_entropy(tf.zeros_like(x), x+0.0001)
    return real_loss + fake_loss
def train_fn(train_dl, epochs, generator, discriminator, lossIn, optimizer_gen, optimizer_dis):
    lmbd = 0.1
    
    for _ in range(epochs):
        for k, (x, y) in enumerate(train_dl):
            with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_dis:
                x = generator(x)
                x_d = discriminator(x)
                y_d = discriminator(y)
                lossD = tf.math.reduce_mean(lossd(x_d,y_d))
                lossG = lmbd * tf.math.reduce_mean(lossg(x)) + lossIn(y, x)

            grads_gen = tape_gen.gradient(lossG, generator.trainable_weights)
            grads_dis = tape_dis.gradient(lossD, discriminator.trainable_weights)

            optimizer_gen.apply_gradients(zip(grads_gen, generator.trainable_weights))
            optimizer_dis.apply_gradients(zip(grads_dis, discriminator.trainable_weights))

            if k % 10 == 0:
                print(k, lossD, lossG)
