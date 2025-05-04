import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Paths to training and testing
train_dir = "C:\\Users\\HP\\Desktop\\Claysys\\train"
test_dir = "C:\\Users\\HP\\Desktop\\Claysys\\test"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)
# Load the pre-trained MobileNetV2 model 
base_model = MobileNetV2(
    input_shape=(48, 48, 3),  
    include_top=False,
    weights='imagenet'
)
# Freeze first 100 layers of MobileNetV2
for layer in base_model.layers[:100]:
    layer.trainable = False
# Unfreeze the remaining layers
for layer in base_model.layers[100:]:
    layer.trainable = True

inputs = Input(shape=(48, 48, 1))
x = Conv2D(3, (3, 3), padding='same')(inputs) 
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(7, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=30,
    callbacks=[early_stop]
)

model.save("emotion_model_improved.h5")  #Save the trained model

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid()
plt.show()
