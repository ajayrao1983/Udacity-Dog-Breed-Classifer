# Udacity-Dog-Breed-Classifer

Link to GitHub repository - https://github.com/ajayrao1983/Udacity-Dog-Breed-Classifer

##Libraries Used:
Python:
- keras (version 2.3.1)
- tensorflow
- cv2 
- numpy
- argparse 

R:
- reticulate
- shiny
- shinydashboard


##Folder Structure and Files in Repository
.

|--Saved Models

| |- weights.best.inception.hdf5 # saved trained model weights

|--haarcascades

| |- haarcascade_frontalface_alt.xml #Pre-trained open CV model for Human Face detection

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

|--DogBreedClassifieronShiny # Shiny Web App alternative

| |-- helpers

| | |- dog_breed_classifier.py #This is the main python code that calls the pre-trained model and does the classification

| |-- model

| | |- weights.best.inception.hdf5 #Same saved trained model weights but within the Shiny App framework

| | |- haarcascade_frontalface_alt.xml #Same Pre-trained open CV model for Human Face detection but within the Shiny App framework

| |-- temp #This folder is used to store the uploaded image file temporarily

| |- app.R #This is the shiny app code

|--dog_breed_classifier.py # Command line alternate app implementation

|--dog_app_26Nov28.ipynb # Jupyter notebook if you want to train your classifier or test other classifiers

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
I tried building 3 networks for the dog breed classifier with following accuracy scores:
________________________________________________
| Model                              | Accuracy |
________________________________________________
| CNN from scratch                   |      6%  |
________________________________________________
| Transfer learning with VGG16       |     48%  |
________________________________________________
| Transfer learning with InceptionV3 |     78%  |
________________________________________________

The trained network weights for Inceptionv3 are saved in the 'Saved Models' folder to reuse in the app implementation.

## Analysis

There are a total of 8,351 dog images which have been split into 6,680 images in our training set, 835 in the validation set and 836 in the testing set.
The training set contains 133 dog breeds but it is unbalanced. If each category had the same number of images or observations, one would expect approx.. 50 images for each of the 133 dog breeds in the training set. However, there are some dog breeds that have over 70 images while some have less than 30. This could impact model performance on some of these dog breeds with lower samples to train on. 

Another point to keep in mind is that, this is a more granular classification problem. We are no longer dealing with having to detect differences between species but go deeper and find differences within species. In other words, finding differences between dogs is a harder problem than finding differences between a dog and a cat. 

Some dog breeds can be very similar which makes it harder for the model to differentiate between them. For example, ‘Brittany’ and ‘Welsh Springer Spaniel’ can look very similar and hence would be difficult to tell apart.  
 
Some dog breeds can come in different colors, for example Yellow, Chocolate and Black Labradors.

These further accentuates the problem, given that there is limited data to work with.

## Conclusion

I was able to build a model using transfer learning on pre-trained InceptionV3 model with an accuracy of 78% vs the goal of 60%. The model does a great job with 103 of 132 dog breeds in the test dataset with an accuracy of 60% or more.

However, it struggles to make predictions for some dog breeds like Greyhound, Black and tan coonhound and Icelandic sheepdog with accuracy scores of 0%. These definitely need to be analyzed further.

Given time and resources, I would have tried the following to improve the model performance generally (not specific to the three dog breeds mentioned above):
•	Increase the training set

•	Augment the data

•	Oversample on dog breeds with less representation (have fewer examples in the training set)

This was an interesting project to me as I was able to bring learnings from Advanced Machine Learning (Convolutional Neural Networks) and Data Scientist Nanodegree (Software/Data Engineering) programs together. By reusing a pre-trained image classification model, I was amazed to have built a model with only 6,680 images that performed with approx.. 80% accuracy.


##App Funcationality in more details

1) If using the command line interface option:

Run the command: python dog_breed_classifer.py <path of the image file>, in the root folder containing 'dog_breed_classifier.py' file

2) If using the Flask Web App option:

Run the command: python main.py, in the 'app' folder

Go to http://0.0.0.0:3001/ or http://localhost:3001/

Upload the image to classify and click 'Submit'

3) If using the Shiny App:
Open app.R in R Studio. Install reticulate, shiny and shinydashboard packages in R.

Assuming you have the necessary python setup as described at the start of this Readme file, you can render the app by clicking 'Run App'
