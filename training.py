from models import *
from traitementimages import *
import tensorflow as tf

B = 5
epochs = 5
loss = tf.keras.losses.MeanSquaredError()
dataset = "Dataset_Test"
a, b = min_dim(dataset)
dim_entrainement=(a//4,b//4, 3)
print(dim_entrainement)
model = create_model(B, dim_entrainement)
img_entrainement = crop_dataset_upscale(dataset)
img_downscale = resize_dataset(dataset)

img_entrainement = tf.convert_to_tensor(img_entrainement)
img_downscale = tf.convert_to_tensor(img_downscale)

train_model(img_downscale, img_entrainement, epochs, model, loss, batch_size=128)

afficher(img_entrainement[0])
afficher(model.predict([img_downscale[0]])[0])
