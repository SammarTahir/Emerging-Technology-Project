# Sourced from: https://github.com/ianmcloughlin/random-web-app/blob/master/random-wa.py
# Adapted from: https://www.palletsprojects.com/p/flask/
# Run with: env FLASK_APP=random-wa.py flask run

# For creating the web application.
import flask as fl
import keras as kr
import numpy as np
import tensorflow as tf
# Sourced from: https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
# Solution for tensor is not an element of this graph
graph = tf.get_default_graph() 
import base64
# Solution for cv2 https://stackoverflow.com/questions/19876079/cannot-find-module-cv2-when-using-opencv
import cv2
from PIL import Image, ImageOps

# Create the web application.
app = fl.Flask(__name__, template_folder='static')

# Loading the nueral network
model = kr.models.load_model('nerualNetwork.h5')
# Sourced from: https://stackoverflow.com/questions/53391618/tensor-tensorpredictions-softmax0-shape-1000-dtype-float32-is-not-an
model._make_predict_function() 
print('model loaded') # just to keep track in server

# Used for resizing the images from the MNIST
height = 28
width = 28
size = height, width

# Home page
@app.route('/')
def home():
    return fl.render_template('index.html')

# Page for sending image
@app.route('/image', methods=['POST'])
def convertImage():
    # Getting information from user
    encoded = fl.request.values[('imgBase64')]

    # decode the dataURL
    # remove the added part of the url start from the 22 index of the image array
    decoded = base64.b64decode(encoded[22:])

    # Saving the image
    with open('image.png', 'wb') as f:
        f.write(decoded)
    userImage = Image.open("image.png")

    # Resizing the image so it is suitable for the MNIST dataset
    # Sourced from: https://github.com/python-pillow/Pillow/blob/3.0.x/docs/reference/Image.rst
    mnistImage = ImageOps.fit(userImage, size, Image.ANTIALIAS)

    # Saving and loading the new resized images
    mnistImage.save("newImage.png")
    newImage = cv2.imread("newImage.png")

    # Soruced from: https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    # Reshaping and adding to nparray
    grayScaleImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    # Converting to float32 and dividing by 255 for attempted normilization(Does not really impact accuracy of web app)
    grayScaleArray = np.array(grayScaleImage, dtype=np.float32).reshape(1, 784)
    grayScaleArray /= 255

    # setter and getter to return the predicition from the model
    setPrediction = model.predict(grayScaleArray)
    getPrediction = np.array(setPrediction[0])

    # np.argmax returns the highest value ie what should be the same as the digit passed
    predictedNumber = str(np.argmax(getPrediction))
    print(predictedNumber)

    # returns the predicted number to be passed to the .js file
    return predictedNumber


#Main method
app.run()