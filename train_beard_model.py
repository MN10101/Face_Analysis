import tensorflow as tf
from tensorflow.keras import layers, models, optimizers  # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import os

# Paths to the train and test directories
train_dir = 'C:/Users/mnde/Desktop/apps/FaceAnalysisDjango/face_analysis/beard_dataset/train'
test_dir = 'C:/Users/mnde/Desktop/apps/FaceAnalysisDjango/face_analysis/beard_dataset/test'

# Load Pretrained VGG16 Model (Without Fully Connected Layers)
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze Most Layers (Keep Only the Last Few Trainable)
for layer in vgg_conv.layers[:-8]: 
    layer.trainable = False

# Define New Model
model = models.Sequential([
    vgg_conv,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),  
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  
])

# Data Augmentation (For Small Datasets)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
batch_size = 16  # Increase batch size
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary'
)

# Compile Model with Lower Learning Rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.00005),  
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train Model
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // batch_size),
    epochs=50,
    validation_data=test_generator,
    validation_steps=max(1, test_generator.samples // batch_size),
    callbacks=[early_stopping]
)

# Save the trained model to the specified directory
save_dir = 'analyzer/models'
os.makedirs(save_dir, exist_ok=True) 
model_path = os.path.join(save_dir, 'beard_classification_vgg16_finetuned_all_layers.keras')
model.save(model_path)

# Output model accuracy
print(f"Model saved to {model_path}")
print(f"Test accuracy: {history.history['accuracy'][-1]}")
print(f"Test loss: {history.history['loss'][-1]}")
