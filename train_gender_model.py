import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout  # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import os

# Set paths
DATASET_PATH = "C:/Users/mnde/Desktop/apps/FaceAnalysisDjango/face_analysis/dataset"
MODEL_SAVE_PATH = "analyzer/models/gender_classification_vgg16_finetuned_all_layers.keras"

# Load the VGG16 model (pretrained on ImageNet)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Unfreeze all layers of the base model
for layer in base_model.layers:
    layer.trainable = True

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  
])

# Compile the model with a slightly higher learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,  
    width_shift_range=0.3, 
    height_shift_range=0.3,  
    shear_range=0.3,  
    zoom_range=0.3,  
    horizontal_flip=True
)

# Load training data from 'train/' folder
train_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

# Load validation data from 'test/' folder
val_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

# Set up EarlyStopping callback to monitor validation loss
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the model with the fine-tuned layers
model.fit(
    train_generator, 
    validation_data=val_generator, 
    epochs=30,  
    callbacks=[early_stopping]
)

# Save the trained model
model.save(MODEL_SAVE_PATH)

print(f"Model saved to {MODEL_SAVE_PATH}")

# Evaluate the model on the validation set
test_loss, test_accuracy = model.evaluate(val_generator)
print(f"Test accuracy: {test_accuracy}")
print(f"Test loss: {test_loss}")
