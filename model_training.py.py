# model.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Configuration
IMG_SIZE = (48, 48)    # FER uses 48x48 grayscale
BATCH_SIZE = 64
EPOCHS = 25
NUM_CLASSES = 7        # angry, disgust, fear, happy, sad, surprise, neutral

# Data path expectations:
# train_dir/
#   angry/
#   disgust/
#   ...
# valid_dir/
#   angry/
#   ...
# Option: you'll need to prepare/convert dataset into this format or modify code.

train_dir = 'data/train'   # create these folders with images
valid_dir = 'data/valid'

def build_model(input_shape=(48,48,1), num_classes=7):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train():
    # Use ImageDataGenerator for preprocessing; use grayscale mode.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        raise RuntimeError("Please prepare training and validation folders at 'data/train' and 'data/valid' with subfolders for each emotion.")

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    valid_generator = val_datagen.flow_from_directory(
        valid_dir,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=len(train_generator.class_indices))
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator,
        callbacks=[checkpoint, early]
    )

    # Final save (redundant because checkpoint saves best)
    model.save('model.h5')
    print("Saved model to model.h5")

if __name__ == "__main__":
    train()
