
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt

classifier = Sequential()


classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))


classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten())


classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale = 1./255,          # Scaling the pixels values in data
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64,64),
                                                 batch_size = 2,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 2,
                                            class_mode = 'binary')

h = classifier.fit_generator(training_set,
                         samples_per_epoch = 40 ,
                         nb_epoch = 70,
                         validation_data = test_set,
                         nb_val_samples = 8)
                         
                         
plt.figure(figsize=(10, 4))
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')                         

plt.figure(figsize=(10, 4))
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.show()
