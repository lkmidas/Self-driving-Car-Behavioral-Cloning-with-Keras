# Self-driving Car Behavioral Cloning with Keras

## Introduction
This is a team project to build an **end-to-end driving behavioral cloning neural network** based on the one in Udacity Behavioral Cloning project. One big difference between our project and Udacity's is that **our project can learn to drive on both one-lane and two-lane roads**. 

Note that this project was made for `Python 3.7`, `tensorflow 2.3.0` and `Keras 2.4.3`. Running it on another environments may not work as expected.

## Files included
- `model.ipynb`: a Jupyter notebook file to train the model.
- `drive.py`: the script to drive the car on the simulation environment.
- `model.h5`: our pre-trained model.
- `requirements.txt`: all required `pip` packages used in this project.

## Training data
The training data used to train the model are collected from the 3 simulation maps provided by Udacity (you can get them [here](https://github.com/udacity/self-driving-car-sim)) and some other maps from other self-driving competitions. As with most of the deep learning models, especially end-to-end ones, the more data the better. But for demonstration purpose, you can just use the data from Udacity's maps and it will work just fine.

## Preprocessing
### Augmentation
Some augmentation techniques are used in the preprocessing phase of the training process, they are:
- Using left and right camera images (only for the data collected from Udacity's maps, other data sources may only contain central camera images).
- Randomly translate (shift) the images horizontally and vertically.
- Randomly modify the brightness of the images.
- Randomly adding shadows to the images.
- DON'T use random flipping augmentation method for this model, it's because we also want to train the car to drive on the correct lane on a two-lane road, flipping the image will make the car driving on the wrong lane.

The detailed implementation of the augmentation phase can be found in `model.ipynb`.

### More preprocessing
Before feeding the augmented images into the training process, we also do some further preprocessing to fit the images into the requirements of the model and improves overall performance:
- The collected images are 320x160, we crop 60 pixels from the top and 25 pixels from the bottom to remove the sky and the front part of the car from the images.
- Then, we resize the images to 64x64, which is the required image size for the model.
- Convert the color space to YUV (as suggested in NVIDIA's PilotNet paper).
- Add some blur to the image to remove noises.

## Training model
### Neural network architecture
The neural network architecture we used in this project is a quite simple one:
- Start off with a `Lambda` layer to normalize the pixels to `[-0.5, 0.5]`.
- Then, a `Conv2D` with 3 1x1 filter. Initially, we didn't use this layer, but since we add it, the performance improved significantly. The purpose of this layer is for the model to learn the most suitable color space to work with.
- Next are 3 convolutional blocks, each convolutional blocks contain two `Conv2D` layers which comprise of 32, 64 and 128 3x3 filters, respectively. The convolutional layers are followed by a 2x2 `MaxPooling2D` layer, and a `Dropout` layer for regularization.
- Finally, there are 3 fully-connected `Dense` layers with 512, 64 and 16 neurons. 

### Training hyperparameters
These are the final hyperparameters that we used after tuning for about 30-40 iterations of the training process (yours may be different):
- Activation function: **ELU**
- Optimizer: **Adam** (with default Adam's hyperparameters)
- Learning rate: **0.0002**
- Dropout rate: **0.35**
- Batch size: **32**
- Steps per epoch: **2000**
- **Early stopping** to select the best model, usually stop at around the 70th epoch.

For the best model we have trained (which is included in this repository), we achieved a very low loss, very close to **0.03**.

## Contributors
- [Khac Minh Dang Le](https://github.com/LKMDang)
- [Tra Yen Nhu Phan](https://github.com/alexptyn)
- Minh Thu Nguyen
