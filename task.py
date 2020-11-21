from tensorflow.keras import layers, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

weight_path = 'inception_v3.h5'
np.random.seed(48)

# training, validating and testing data generators
training_datagen = ImageDataGenerator(rescale=1./255)
training_data = training_datagen.flow_from_directory(
        r'hand signs/training',
        target_size=(150, 150),
        batch_size=50,
        class_mode='binary'
    )

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_data = validation_datagen.flow_from_directory(
        r'hand signs/validation',
        target_size=(150, 150),
        batch_size=50,
        class_mode='binary'
    )

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
        r'hand signs/test',
        target_size=(150,150),
        batch_size=1,
        class_mode='binary',
    )

# Using the inception model
pretrained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
pretrained_model.load_weights(weight_path)

# freeze the layers
for layer in pretrained_model.layers:
    layer.trainable = False

last_layer = pretrained_model.get_layer('mixed7')
last_output = last_layer.output

# Creating our shallow network
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='tanh')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='tanh')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation='tanh')(x)
x = layers.Dense(64, activation='tanh')(x)
x = layers.Dense(10, activation='softmax')(x)

model = Model(pretrained_model.input, x)

model.compile(
        optimizer=Adam(lr=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Uncomment the ModelCheckpoint for saving the best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4),
    # ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
]

history = model.fit(
        training_data,
        steps_per_epoch=8,
        epochs=50,
        validation_data=validation_data,
        validation_steps=4,
        callbacks=callbacks,
        verbose=2,
    )

# Evaluation from test data
results = model.evaluate(test_data, batch_size=50, verbose=0)
print(f'\nEvaluation Results: accuracy: {round(results[1], 4)} and loss: {round(results[0], 4)}')

# visualization of accracy and loss
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', marker='.', label='accuraacy')
plt.plot(epochs, val_acc, 'b', marker='.', label='validation accuracy')
# plt.plot(epochs, loss, 'b', label='loss')
# plt.plot(epochs, val_loss, 'y', label='validation loss')
plt.legend()
plt.show()
