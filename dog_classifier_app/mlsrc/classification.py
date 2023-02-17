from keras.utils import get_file
from keras.models import load_model
from .extract_bottleneck_features import *
import numpy as np
from .image_processing import *
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from pathlib import Path


# define function to load train, test, and validation datasets
def load_dataset(path: Path):
    """Load a single dataset on a given path

    Args:
        path (Path): Path to a dataset

    Returns:
        list: Returns list with files and targets in the dataset
    """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# load train, test, and validation datasets
def load_dog_datasets(path: Path, print_info=False):
    """Load all datasets related to dog_images. This function is used for development purpose
    and can print and check if all data is loaded, and provides all data needed for training the model.

    Args:
        path (Path): root directory containing train, valid and test sub-folders.
        print_info (bool, optional): Flag for printing information about loaded datasets. Defaults to False.

    Returns:
        list: Returns list of train, validation and test datasets for dog images.
    """
    assert isinstance(path, Path), "Please specify **path** as pathlib.Path object"
    train_files, train_targets = load_dataset(  path / 'train')
    valid_files, valid_targets = load_dataset( path / 'valid')
    test_files, test_targets = load_dataset( path / 'test')


    if print_info:
        # load list of dog names
        dog_names = [item.as_posix().split('/')[-1].split(".")[-1] for item in sorted(path.glob('./train/*/'))]

        # print statistics about the dataset
        print('There are %d total dog categories.' % len(dog_names))
        print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
        print('There are %d training dog images.' % len(train_files))
        print('There are %d validation dog images.' % len(valid_files))
        print('There are %d test dog images.'% len(test_files))

    return train_targets, valid_targets, test_targets

def load_dog_categories(path: Path):
    """Loads categories in dog images. This functions should be called once session starts, since categories should be available.

    Args:
        path (Path): Path to categories used in training.

    Returns:
        list: List of dog names (categories).
    """
    assert isinstance(path, Path), "Not valid __path__. Please provide __path__ as pathlib.Path object!"
        # load list of dog names
    dog_names = [item.as_posix().split('/')[-1].split(".")[-1] for item in sorted(path.glob('./train/*/'))]
    return dog_names

DOG_CAT_PATH = Path(__file__).parent / "data" / "dog_images"
DOG_NAMES = load_dog_categories(DOG_CAT_PATH)

# This should be stored in some json file, but dict will do for now...
RESNET_URL = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz'
VGG19_URL = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz'
INCEPTION_URL = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz'
XCEPTION_URL = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz'

network_npz = {
    'Resnet50': RESNET_URL,
    'VGG19': VGG19_URL,
    'Inception': INCEPTION_URL,
    'Xception': XCEPTION_URL
}

network_bottleneck = {
    'Resnet50': lambda x: extract_Resnet50(x),
    'VGG19': lambda x: extract_VGG19(x),
    'Inception': lambda x: extract_InceptionV3(x),
    'Xception': lambda x: extract_Xception(x)
}

# How to load npz from servers: https://www.tensorflow.org/tutorials/load_data/numpy
def load_npz_from_url(network):
    """
    Downloads npz file from Udacity's AWS server and creates train, validation and test data.
    
    Args: network - key value to dict holding the URL
    
    Returns: list (train_data, valid_data, test_data): np.array for train, validation and test data
    
    """
    
    assert network in network_npz.keys(), 'Cannot figure out the URL!'
    
    path = get_file(f'tmp_Dog{network}_Data.npz', network_npz[network])
    with np.load(path) as data:
        train_data = data['train']
        valid_data = data['valid']
        test_data = data['test']
    
    return [train_data, valid_data, test_data]

def train_model(network, epochs=20, data_path = Path(__file__).parent / "data/dog_images"):
    """Use transfer learning to train the model faster using pre-trained bottleneck features from other CNN.
    This function is used only in notebook for training.

    Args:
        network (string): Bottleneck features network to use
        epochs (int, optional): Number of epochs to use. Defaults to 20.
        data_path (Path, optional): Path to data files. Defaults to Path(__file__).parent/"data/dog_images".

    Returns:
        list: Returns trained model and test data for convenient post-testing.
    """
     
    train_data, valid_data, test_data = load_npz_from_url(network)
    train_targets, valid_targets, _ = load_dog_datasets(data_path)
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(133, activation='softmax'))

    
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    # Training step:
    checkpointer = ModelCheckpoint(filepath=f'saved_models/weights.best.{network}.hdf5', 
                               verbose=1, save_best_only=True)

    model.fit(train_data, train_targets, 
              validation_data=(valid_data, valid_targets),
              epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=2)
    
    
    print("--------------------------")
    print("SAVING MODEL")
    model.save(f'./saved_models/model.dev.{network}.hdf5')
    
    
    return model, test_data

def test_model(model, test_data, test_targets):
    """Utility function to test the model. Returns the accuracy of the given model for the input data.

    Args:
        model (TensorFlow model): CNN model to be tested.
        test_data (list): List of data to be tested
        test_targets (list): Test labels.

    Returns:
        float: Returns test accuracy
    """
    # get index of predicted dog breed for each image in test set
    model_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_data]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(model_predictions)==np.argmax(test_targets, axis=1))/len(model_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)
    return test_accuracy


def get_model(network, dev=True, path_to_models=Path(__file__).parent / "saved_models"):
    """Loads the saved model to be used.

    Args:
        network (string): Bottleneck network used for training.
        dev (bool, optional): Flag to tell if there is dev in name of model. Defaults to True.
        path_to_models (Path, optional): Path to model. Defaults to Path(__file__).parent/"saved_models".

    Returns:
        tf.model: Tensorflow model.
    """
    assert network in network_bottleneck.keys(), 'Cannot find extract function!'
    # TODO: make this more general, since this is dev at the moment!!!
    model = load_model( path_to_models /  f'model.dev.{network}.hdf5')
    return model
    

def predict_dog_breed(model, network, img_path):
    """Core function for predicting dog breed. It takes model, network for bottleneck features, path to image to predict the breed of dog in the image. 
    This function assumes that there is dog in the image.

    Args:
        model (Tensorflow model): CNN model used for classification/prediction.
        network (string): Network of bottleneck features.
        img_path (Path): Path to image
        dog_categories_path (Path, optional): Path where image labels for dog categories are stored. Defaults to Path(__file__).parent/"data"/"dog_images".

    Returns:
        _type_: _description_
    """
    assert network in network_bottleneck.keys(), 'Cannot find extract function!'
    # Extract bottleneck features
    bottleneck_feature = network_bottleneck[network](path_to_tensor(img_path))    
    
    # Propagate the feauters
    predicted_probs = model.predict(bottleneck_feature)
    sorted_idx = np.argsort(predicted_probs)
    dog_names = DOG_NAMES

    _N_FIRST_RET = 5

    sel_idx = sorted_idx
    kks = np.array(dog_names)[sorted_idx][0]
    vvs = np.take_along_axis(predicted_probs, sorted_idx, axis=-1)[0]
    print(kks,vvs)

    return dog_names[np.argmax(predicted_probs)], dict(zip( reversed(kks[-_N_FIRST_RET:]), [float(item) for item in reversed(vvs[-_N_FIRST_RET:])]))


def dog_breed_clasifier(model, img_path: Path, network = 'Xception') -> str:
    """Main access to the dog breed classifier. This method performs actual classification.
    Using the desired network, the pre-trained model is loaded and used for classification.
    This function is result from the main notebook on Udacity capstone project workspace.

    Args:
        model (tf.Model): TensorFlow(Keras) model 
        img_path (Path): Path to the file to be classified.
        network (str, optional): Network. Defaults to 'Xception'.

    Returns:
        str: Returns the dog breed as a string
    """

    #NOTE --> this file differs from the one on Udacity website!

    # Sometimes, images from iPhone are not converted, and OpenCV has issues...
    try:
        _, _ishuman = face_detector(img_path=img_path)
        _isdog = dog_detector(img_path=img_path)
    except:
        return ["PROBLEM WITH IMAGE, PLEASE USE JPG!", "ERROR"]
    
    # img = plt.imread(img_path)
    # display the image, along with bounding box
    # print('Your input:')
    # plt.imshow(img)
    
    publish_string = ""
    _breed = None
    if (_ishuman == False) and (_isdog == False):
        publish_string = publish_string + "I could not detect humans nor dogs in this image!"
        return [publish_string, _breed, {100: None}]
    
    # Do predicition
    _breed, _probs_dict = predict_dog_breed(model=model, network=network, img_path=img_path)
    if _ishuman and _isdog:
        publish_string = publish_string + "Hmmmmm... It looks like there are both humans and dogs in this image!"
    elif _ishuman:
        publish_string = publish_string + "Hmmmmm... It looks like this is image containing human(s)!"
    elif _isdog:
        publish_string = publish_string + "Only dogs in this image!"

    return [publish_string, _breed, _probs_dict]


if __name__ == '__main__':
    # Testing features
    
    # In case that only cells in step 5 are run, do the imports here
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from keras.layers import Dropout, Flatten, Dense
    from keras.models import Sequential
    
    # with active_session():
    #     # do long-running work here

    from keras.callbacks import ModelCheckpoint 

    # acc_dict = dict()
    # for kk,val in network_npz.items():

    #     tmp_model, test_data = train_model(kk)
    #     acc_dict[kk] = test_model(tmp_model, test_data, test_targets)
    data_path = Path(__file__).parent / "data" / "dog_images"
    
    cats = load_dog_categories(data_path)

    model = get_model(network='Inception')
    predicted, pprobs = predict_dog_breed(model, "Inception", img_path = Path(__file__).parent / "data/dog_images/test/004.Akita/Akita_00244.jpg")
    print("DONE")
