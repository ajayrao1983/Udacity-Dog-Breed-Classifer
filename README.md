# Udacity-Dog-Breed-Classifer

Link to GitHub repository - https://github.com/ajayrao1983/Udacity-Dog-Breed-Classifer

##Libraries Used:

keras (version 2.3.1)

cv2 

numpy

argparse 


##Folder Structure and Files in Repository
.

|--Saved Models

| |- weights.best.inception.hdf5 # saved trained model weights

|--haarcascades

| |- haarcascade_frontalface_alt.xml # Has the categories for each message in the 'disaster_messages.csv' file. Categories are what we are trying to predict.

|--images # collection of images to test the app

| |- cat1.jpg

| |- cat2.jpg

| |- dog1.jpg

| |- dog2.jpg

| |- dog3.jpg

| |- dog4.jpg

| |- human1.jpg

| |- human2.jpg

| |- human3.jpg

| |- human4.jpg

|--app # Flask web app alternative

| |- static

| | |- uploads # Images uploaded in the Flask webapp are stored in this folder 

| |- templates

| |- upload.html # The html template for Flask web app

| |- main.py # Flask web app

|--dog_breed_classifier.py # Command line alternate app implementation

|--dog_app_26Nov21.ipynb # Jupyter notebook if you want to train your classifier or test other classifiers

|--README.md

##Project Description

We are designing an algorithm that takes an image as input and,
1) if it detects a dog, provides an estimate of the dog breed

2) if it detects a human, provides an estimate of the dog breed the human most resembles

3) informs the user if neither a dog or a human is detected

A short summary of the steps taken to train the model is given below but please refer to the python notebook for more details. 
We start with a small set of 6,680 dog images in our training set, out of a total of 8,351 dog images. 
OpenCV's 'Haar feature-based cascade classifier' is used to detect human faces, while a ResNet50 model pre-trained on ImageNet dataset is used to detect dogs in the given image.
We then pull a InceptionV3 network pre-trained on ImageNet dataset, and re-purpose it to classify dog breeds if a human or dog image is detected.
We accomplish the transfer learning by removing the last layer of the InceptionV3 network, and appending a layer that is trained using 6,680 dog image datasets to detect dog breeds.

The trained network weights are saved in the 'Saved Models' folder to reuse in the app implementation.

##App Funcationality in more details

1) If using the command line interface option:

Run the command: python dog_breed_classifer.py <path of the image file>, in the root folder containing 'dog_breed_classifier.py' file

2) If using the Flask Web App option:

Run the command: python main.py, in the 'app' folder

Go to http://0.0.0.0:3001/ or http://localhost:3001/

Upload the image to classify and click 'Submit'
