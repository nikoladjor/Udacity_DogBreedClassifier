[//]: # (Image References)

[image1]: ./dog_classifier_app/mlsrc/data/images/sample_dog_output.png "Sample Output"
[image2]: ./dog_classifier_app/mlsrc/data/images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./dog_classifier_app/mlsrc/data/images/vgg16_model_draw.png "VGG16 Model Figure"
[imageHomepage]: ./docs/DogBreedHomePage.png 	"App Home Page"
[imageUpload]: ./docs/DogBreedImageUpload.png 	"App Image Upload"
[imageUsage]: ./docs/DogBreedExample.png		"App Example Usage"


# Udacity Data Science Nanodegree Capstone: Dog Breed Classifier

## Project Description

This is a web application for the Udacity Data science Nanodegree Capstone project. In this project, I built a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, the underlying algorithm will identify an estimate of the dog’s breed.  If supplied an image of a human, the code will identify the resembling dog breed. The goal of the project is that the final deep-learning model has accuracy score larger then 60%. Final model developed using a transfer-learining approach with Xception bottleneck features has accuracy of **>84%**.

Sample output:\
![Sample Output][imageUsage]


## Deep-Learning Pipeline

In order to build the dog breed classifier, all the data for training models and building the final neural network are provided by Udacity. There are three main steps towards the final model build.

1. [Detecting the face in the image](#face-detector)
2. [Detecting the dog in the image](#dog-detector)
3. [Building the convolutional neural network](#transfer-learning)

### Face Detector
To detect faces in the user-supplied image, we use OpenCV's implementation of [Haar feature-based cascade classifiers](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on github, but in the project, Udacity already supplied on of the detectors.

The resulting [detector](./dog_classifier_app/mlsrc/image_processing.py) were tested to detect ~100% of faces, but also yields ~11% of _false-positives_, i.e. in 11% of the cases detects humans in dog images.

### Dog Detector
A pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model is used to detect dogs in images. ResNet-50 model is loaded through Keras, along with weights that have been trained on [ImageNet](https://image-net.org/). The ImageNet is a very large, very popular dataset used for image classification and other vision tasks.

Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.

Since TensorFlow is used as a back-end for Keras during the develeopment, it is important to perform data pre-processing, which boils down to reshaping the data. Data pre-processing, as well as details about loading the model is given in the jupyter-notebook which is used for model develeopment on the AWS instance provided by Udacity.

### [Transfer Learning](https://keras.io/guides/transfer_learning/)
In order to achieve the required high accuracy while keeping the training time as short as possible transfer-learning approach was used. The transfer-learning is a machine learning workflow in which a pre-trained neural network (and its learned features) is used as part of the larger network. In this project, we use Xception features trained on ImageNet, which yields accuarcy of **>84%**.  


## Project Instructions

### Examples

After the server start, the homepage should look like this:\
![HomePage][imageHomepage]

After creating a session, an user can upload images:
![Upload][imageUpload]


Finally, after each image upload, CNN model is called to predict the dog breed if human or dog are detected in the image:
![Result][imageUsage] 
Alongside the predicted breed, top 5 predicted probabilities are shown.


### Installation

#### Standard install

The app is developed using `python 3.10.7` and all packages can be installed using 
```shell
git clone https://github.com/nikoladjor/Udacity_DogBreedClassifier
pip install requirements.txt
```
After installation, the app can be started as using standard flask command:
```python
flask --app dog_classifier_app/app.py --debug run
```
> **NOTE**: during the testing on macOS, I had an issue with SSL certificates that are needed to download the ResNet50. This is why I also decided to publish the app with Docker as well.


#### Installation using Docker
This web-app can also be run using Docker. If new to Docker, please follow the installation and building app [tutorial](https://docs.docker.com/language/python/build-images/).

Assuming that Docker is up and running, try to build and run the container 
```bash
docker build -t test_me:latest .
```



>**NOTE**: The original version of project that can be cloned from Udacity GitHub repository can be found in [docs folder](./docs/README_OLD.md). The training jupyter notebook is using virtual environment pre-set on AWS and is different from the environment used for the app developement. 