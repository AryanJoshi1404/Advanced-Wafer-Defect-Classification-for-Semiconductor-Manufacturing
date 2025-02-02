# the code given below consists of both the data augmentation part as well as the model training part
# all written in a single code 
   
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# just for testing to see if tensorflow is using GPU or not 

# print(tf.__version__)
# gpus = tf.config.list_physical_devices('GPU')
# if not gpus:
#     print("No GPU detected.")
# else:
#     print(f"Available GPUs: {gpus}")


# Data Augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation and test data, only rescaling
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
train_data = train_datagen.flow_from_directory(
    "data/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_test_datagen.flow_from_directory(
    "data/validation",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = val_test_datagen.flow_from_directory(
    "data/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load the VGG16 architecture with pretrained weights
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze VGG16 layers to retain the pretrained features
for layer in vgg16.layers:
    layer.trainable = False

# Create the model and add layers
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(train_data.num_classes, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set up Early Stopping with validation loss
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training with validation data
history = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data,
    callbacks=[early_stopping]
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

test_data.reset()
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


