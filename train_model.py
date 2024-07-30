import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D # type: ignore
import joblib

# Parameters
batch_size = 32
img_height = 150
img_width = 150
epochs = 10

# Data Preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=40,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2  # 20% of data will be used for validation
)

# Training Data Generator
train_generator = train_datagen.flow_from_directory(
    'dog_breeds',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation Data Generator
validation_generator = train_datagen.flow_from_directory(
    'dog_breeds',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Model Definition
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the Model
model.save('dog_breed_model.h5')

# Save Class Labels
class_labels = list(train_generator.class_indices.keys())
with open('class_labels.pkl', 'wb') as f:
    joblib.dump(class_labels, f)

# Evaluate the Model
eval_result = model.evaluate(validation_generator)
print(f"Validation Loss: {eval_result[0]}")
print(f"Validation Accuracy: {eval_result[1] * 100:.2f}%")
