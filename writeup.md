# **Traffic Sign Recognition** 

[//]: # (Image References)

[learning]: ./images/learning.png "Validation and training set accuracies during learning"
[class_examples]: ./images/class_examples.png "An example of each of the training classes"
[augmented_examples]: ./images/augmented_images.png "Examples of some augmented images"
[training_distribution]: ./images/training_distribution.png "Distribution of classes in the training data"
[new_images]: ./images/new_images.png "New Traffic Signs"
[softmax]: ./images/softmax.png "Softmax probabilities"
[comparison]: ./images/comparison_2_epochs.png "Classification after 2 epochs"

# Dataset Exploration
## Dataset Summary
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

## Exploratory Visualization
The notebook contains plots of the distribution of classes for the training, validation and test data sets.
The distribution of classes in the training data set is shown below:
![Distribution of classes in training data set][training_distribution]
From this plot we can see that all classes have at least 200 examples, but that some classes include up to 10 times more instances.
This skewed distribution will probably lead to a bias in the trained model such that it is more likely to predict an input image as beloning to one of the
over-represented classes. I assume, however, that the distribution of the classes is somewhat in line with the actual frequency of the traffic signs, so that
this bias can actually be beneficial.

A random image from all of the classes in the training data set is included below:
![An example of each of the training classes][class_examples]

# Model Architecture
## Preprocessing
The images in all of the sets are first normalized.
The `skimage.exposure` function `equalize_hist()` results in better model performance than the proposed `(pixel - 128) / 128` approach.
The color channels are kept as the information about the image of a sign might be beneficial.

For training, the images from the training data set are then augmented using a number of operations:
rotation, scaling and adding noise.
The scaling is constrained to a factor of between 0.9 and 1.3, while the rotation is constrained to -10 to +10 degrees.
The noise is of mean zero and variance 0.001.
An example of some augmented images is shown below:
![Examples of some augmented images][augmented_examples]

## Model Architecture
The model employed is a modified LeNet architecture.
The modifications consist in using three input channels (r, g, b) instead of a single one, changes in the number of convolutional filters, the number of fully connected neurons and the use of dropout.
The depth of the two convolutional layers was multiplied by four.
The number of neurons in the fully connected layers was doubled.
Dropout with a value of 0.75 was used in the fully connected layers.

## Model Training
The original training data set was concatenated with two times the number of augmented images (this was the maximum amount of images that
the memory on the training machine allowed).
The model was then trained with the Adam optimizer for 50 epochs with a batch size of 128 and a learning rate of 0.001.
During training, the validation accuracy increased to over .96, while the training accuracy increased to over .98.

## Solution Approach
The LeNet base was chosen because of the similarity of recognizing digits and traffic signs.
Both domains contain only a limited number of instances (e.g. 10 or 43).
Traffic signs are standardized, so that a network of limited complexity should be able to learn to recognize them.

The LeNet worked rather well out-of-the box.
The most significant increase in performance was gained by changing the normalization to the `equalize_hist()` approach.

Experiments with a very high dropout rate of e.g. .35 were very interesting because of the heightened robustness, but also
required quite a long time to train.

The final solution trains rather quickly, but still gives pretty good results on the test set and the custom images sourced from the web.
![Validation and training set accuracies during learning][learning]

The performance of the final network on the test dataset is above .95.

# Web Images

## About the new images
The traffic signs below were downloaded from the web, roughly cropped and resized to 32x32 pixels.
![New traffic signs from the web][new_images]

The "Beware of ice/snow" sign is tilted slightly, but the training images were augmented in the same way and so the network should be able to recognize this sign.

The "Speed limit 30km/h" sign is translated up from the center and contains additional writing that was not in the training images, so out of all the additional images, this one is probably the hardest for the network to classify.

The other signs are nice examples of their classes and should be recognized by the network.

## Performance on the new images
The network recognizes all web images, which is to be expected given the networks performance on the validation and test image sets.

## Model certainty
As can be seen in the plot below, the network is very certain of its predictions.
![Softmax predictions on web images][softmax]

When testing the model before it is converged (e.g. at only 2 epochs), the probabilities here are still much more distributed.
![Classification results after 2 epochs][comparison]

