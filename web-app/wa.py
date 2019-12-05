# Sourced from: https://github.com/ianmcloughlin/random-web-app/blob/master/random-wa.py
# Adapted from: https://www.palletsprojects.com/p/flask/
# Run with: env FLASK_APP=random-wa.py flask run

# For creating the web application.
import flask as fl
import numpy as np
import base64
import cv2
from PIL import Image, ImageOps

# Used to load the neural network
from keras.models import load_model

# Create the web application.
app = fl.Flask(__name__)

# Loading the nueral network in the code folder
model = load_model('Code/digit_reader.h5')

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
    # Converting the new image to grayscale, reshaping and adding to nparray
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
if __name__ == "__main__":  
    #Run the app.
    app.run()
