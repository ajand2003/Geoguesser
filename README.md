# geoguess-ml
# Intro
This project involves the ability to discern the location of an image using machine learning. There has been a decent amount of work done in this field. In 2016, Google developed an AI called Planet which aimed to recognize the location of a photo anywhere in the world. At the University of Colorado Boulder, researchers aimed to create an AI which could determine the location of a photo in the continental U.S states. However, this area of ML is far from being perfected. The google Planet AI could only detect location with a country-accuracy of 28%. 
# Motivation
The motivation of this project is being able to geotag images without any additional information other than the pixels in the image. The ability to do this could have widespread uses, from tagging photos on social media sites, to cheating in the popular game Geoguessr. 
# Algorithms and Methods
For the proposed dataset, google street api will be used to generate random street images across the 10 selected countries. Around 35000 images will be generated to ensure the proposed model will have enough data to hypertune parameters and ensure accuracy with the validation set in subsequent trials. The dataset will be the input for a convolutional neural network implemented through PyTorch libraries/packages. We will normalize the data and then pass it through a CNN with convolutional, ReLU, max pooling, and fully connected layers. Finally, we will use a cross entropy loss function and optimize with gradient descent.
![general CNN architecture](https://i0.wp.com/developersbreach.com/wp-content/uploads/2020/08/cnn_banner.png?fit=1200%2C564&ssl=1)
*Similar CNN Architecture* [source](https://developersbreach.com/convolution-neural-network-deep-learning/)
# Discussion and Expected Results
There will be a few constraints put on our results. We will not consider the exact location of the image. Instead, we will consider if the correct country was chosen. We will also only consider 10 countries in our algorithm, instead of considering all possible countries in the world. In other predictive uses of convolutional neural networks, well-developed CNNs have been able to outperform humans by around 33 percent*. Accounting for the unique characteristics of our project in addition to knowledge/time constraints, we hypothesize that our algorithm will provide correct predictions at a rate comparable to an educated human. 
As a measure of success, we will compare the probability that our algorithm chooses the right location to the probability that a participant chooses the right location. In order to quantify the overall accuracy of our model, we will use an accuracy score function which computes what percentage of predictions are correct. Our algorithm will be successful if it produces correct guesses at a rate greater than or equal to a participant.
