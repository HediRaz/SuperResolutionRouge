from models import create_model, create_model_D, train_fn
from utils_images import CustomDataGen, create_dataset
import tensorflow as tf
import numpy as np
from utils_test import test_model
from VGG16 import loss_gg


EPOCHS = 1
BATCH_SIZE = 16
train_folder_path = "Dataset/imagenet/train_processed_128"
val_folder_path = "Dataset/imagenet/val_processed_128"

B = 5
# loss = tf.keras.losses.MeanSquaredError()
loss = loss_gg
optimizer = tf.keras.optimizers.Adam()

train_dataset = create_dataset(train_folder_path, batch_size=BATCH_SIZE, input_size=(128, 128, 3), shuffle=True)
val_dataset = create_dataset(val_folder_path, batch_size=BATCH_SIZE, input_size=(128, 128, 3), shuffle=True)

generator = create_model(B, (128//4, 128//4, 3))
discriminator = create_model_D((128, 128, 3))

generator.build((None, None, None, 3))
discriminator.build((None, 128, 128, 3))
psnr = lambda x, y: tf.image.psnr(x, y, 1)

train_fn(train_dataset, EPOCHS, generator, discriminator, optimizer, psnr)
model.save_weights("SavedModels/generator")

print("Test")
test_images = np.array([a[1][0].numpy() for a in val_dataset])
test_model(model, test_images)
