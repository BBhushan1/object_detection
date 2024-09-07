from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os


os.makedirs('checkpoint', exist_ok=True)


train_datagen = ImageDataGenerator(rescale=1/255.,
                                   horizontal_flip=True,
                                   shear_range=0.2,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2)

train = train_datagen.flow_from_directory('Data/Train',
                                          batch_size=32,
                                          target_size=(256, 256),  
                                          class_mode='binary',
                                          shuffle=True,
                                          interpolation='nearest')

test_datagen = ImageDataGenerator(rescale=1/255.)

test = test_datagen.flow_from_directory('Data/Test',
                                        batch_size=32,
                                        target_size=(256, 256),  
                                        class_mode='binary',
                                        shuffle=True,
                                        interpolation='nearest')

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3))) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, mode='max', verbose=1, min_lr=0.001, factor=0.1)
model_checkpoint = ModelCheckpoint('checkpoint/model.keras',  
                                   monitor='val_accuracy',
                                   mode='max',
                                   save_best_only=True,
                                   verbose=1)  

early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=90)

# Model training
history = model.fit(train,
                    epochs=50,
                    batch_size=32,
                    validation_data=test,
                    callbacks=[reduce_lr, model_checkpoint, early_stopping])

plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], 'r', label='train loss')
plt.plot(history.history['val_loss'], 'b', label='test loss')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.legend()

plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], 'r', label='train accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='test accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Graph')
plt.legend()


model.save('ModelE.h5') 

loaded_model = load_model('ModelE.h5')  


loss, acc = loaded_model.evaluate(test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)
