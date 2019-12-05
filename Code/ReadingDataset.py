# This is used to make a nueral network
import keras as kr
# This is used to plot data
import numpy as np
import matplotlib.pyplot as plt
# This is used to upzip the files
# Adapted from: https://docs.python.org/3/library/gzip.html
import gzip

# Importing files from the MNIST website
# The data from these files will be used to make the nerual network
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()

with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()

# Laoding in the files
(train_img, train_lbl), (test_img, test_lbl) = kr.datasets.mnist.load_data()


# Scaling the data 
train_img = train_img.reshape(60000, 784)
test_img = test_img.reshape(10000, 784)

# Dividing the img by 225 for scaling
train_img = train_img.astype('float32')
test_img = test_img.astype('float32')
train_img = train_img/255
test_img = test_img/255 

train_lbl = kr.utils.np_utils.to_categorical(train_lbl, 10)
test_lbl = kr.utils.np_utils.to_categorical(test_lbl, 10)

# This is a for loop for the images
for i in range(50):
    plt.subplot(1,50,i+1)
    
    # This shows the image
    plt.imshow(train_img[i].reshape(28,28), cmap='gray', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    # plt.show()

# This is creating the neural netwrok by using the models import from keras
print("Creating model")
model = kr.models.Sequential()

print("Sequential model created")
print("Adding layers to model...")

# Start a neural network, building it by layers
# Use input_shape=(28,28) for unflattened data
model.add(kr.layers.Dense(392, activation='relu', input_shape=(784,)))
model.add(kr.layers.Dense(392, activation='relu'))

# This is to stop overfilling 
model.add(kr.layers.Dropout(0.2))

# This is the final layer and finishes the nueral network 
# ***For notebook -> The Adam optimization algorithm is an extension to stochastic gradient descent 
# that has recently seen broader adoption for deep learning applications in computer vision and natural language processing***
model.add(kr.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# This is training the nueral network 
# ***For notebook -> epoches meeans the amount of times the test is carried out*** 
history = model.fit(train_img, train_lbl, batch_size=50, epochs=5, verbose=1, validation_data=(test_img, test_lbl))

# This shows the accuracy of the nueral network
score = model.evaluate(train_img, train_lbl, verbose=0)
print('Test cross-entropy loss: %0.9f' % score[0])
print('Test accuracy: %0.9f' % score[1])

# This is plotting the loss
plt.figure(1, figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.show()

# This is plotting the accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# This loads and saves the network
model.save('digit_reader.h5')
loadedModel = kr.models.load_model('digit_reader.h5')

# ***For Notebook -> Add this for predicted number***
plt.imshow(test_img[77].reshape(28, 28), cmap="gray")
plt.show()

# This is given a test image and seen if it is able to load the correct number
print(loadedModel.predict(test_img[77:78]), "\nCaluclated Number: ", np.argmax(loadedModel.predict(test_img[77:78])))