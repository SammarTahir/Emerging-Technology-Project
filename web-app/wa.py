# Adapted from: https://www.palletsprojects.com/p/flask/
# Run with: env FLASK_APP=random-wa.py flask run

# For creating the web application.
import flask as fl
# For generating random numbers.
import numpy as np

# Create the web application.
app = fl.Flask(__name__)

# Add a route for the web page.
@app.route('/')
def home():
  return app.send_static_file('index.html')

# Add a route for generating random numbers using POST data.
@app.route('/random', methods=['GET', 'POST'])
def random():
  # Find out how many numbers the user asked to generate.
  howmany = int(fl.request.values.get("noofnos", "1"))
  # Generate and respond with the numbers.
  return {"randomNos": np.random.random(howmany).tolist()}#