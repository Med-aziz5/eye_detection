import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from preprocess import load_dataset

# ---------------------------
# Load dataset
# ---------------------------
X_train, X_val, y_train, y_val = load_dataset()

# Resize images to 224x224 for VGG16
IMG_SIZE = 224
X_train_resized = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in X_train])
X_val_resized = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in X_val])

# ---------------------------
# Build VGG16 model

# ---------------------------
input_shape = (IMG_SIZE, IMG_SIZE, 3)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)  # 3 classes: normal, sleep, yawn

model = Model(inputs=base_model.input, outputs=predictions)

# ---------------------------
# Compile model
# ---------------------------
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------
# Train model
# ---------------------------
history = model.fit(
    X_train_resized, y_train,
    validation_data=(X_val_resized, y_val),
    epochs=10,
    batch_size=32
)

# ---------------------------
# Optional: Fine-tuning
# ---------------------------
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    X_train_resized, y_train,
    validation_data=(X_val_resized, y_val),
    epochs=5,
    batch_size=32
)
