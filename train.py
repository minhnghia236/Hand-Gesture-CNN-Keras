from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import normalize

train_datagen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
)
                        
validation_datagen = ImageDataGenerator(
            rescale = 1./255,
)

train_generator = train_datagen.flow_from_directory('images/train/',
                                        target_size=(64, 64),
                                        batch_size=32,
                                        color_mode='grayscale',
                                        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory('images/validation/',
                                        target_size=(64, 64),
                                        batch_size=32,
                                        color_mode='grayscale',
                                        class_mode='categorical')

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer=Adam(learning_rate=0.001),
             metrics=['accuracy'])

model.fit(train_generator, steps_per_epoch=10, epochs=100, verbose=1, validation_data=validation_generator)

model.save('model.h5')

print(train_generator.class_indices)