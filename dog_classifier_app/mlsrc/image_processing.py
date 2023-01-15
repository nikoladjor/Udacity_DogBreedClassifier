import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from pathlib import Path
import os

from keras.applications.resnet import ResNet50

# Defining model as a constant here
_RESNET_MODEL = ResNet50(weights='imagenet')
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# FOR TESTING PURPOSES
HUMAN_FILES = np.array(glob(os.fspath(Path(__file__).parent) + "/data/lfw/lfw/*/*"))



# returns "True" if face is detected in image stored at img_path
def face_detector(img_path:Path, filename=None, plot_faces=False):
    """
    Returns number of faces and square around the detected faces if face is detected in image stored at img_path.
    If not (0, False) is returned
    
    Parameters:
        img_path(string) : A string path to the image
        plot_faces(boolean): Plot detected faces or not. Used for offline testing
        filename(string or None): name of the image file, used for saving the image+faces if provided
    Returns:
        boolean: if the face is detected in the image
    """
    img = cv2.imread(img_path.as_posix())
    p = img_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    if len(faces)>0:
        # get bounding box for each detected face
        for (x,y,w,h) in faces:
        # add bounding box to color image
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        # convert BGR image to RGB for plotting
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if plot_faces:
            # display the image, along with bounding box
            plt.imshow(cv_rgb)
            plt.show()
        
        else:
            # Store the file with faces detected
            if not filename:
                # filename is not supplied, just return values, since it is probably for test pupose...
                return len(faces), True
            save_path = p.parent / f'{filename}'
            plt.imsave(arr=cv_rgb, fname=os.fspath(save_path))
            # cv2.imwrite(img=cv_rgb, filename=os.fspath(save_path))
            return len(faces), save_path
    else:
        return 0, False

from keras.preprocessing import image
from keras.utils.image_utils import load_img, img_to_array
from tqdm import tqdm
from keras.applications.resnet import preprocess_input, decode_predictions


def path_to_tensor(img_path):
    """Loads RGB image as PIL.Image.Image typ

    Args:
        img_path (str): Path to image file

    Returns:
        4D tensor: 4D tensort with shape (1, 224, 224, 3)
    """
    # loads RGB image as PIL.Image.Image type
    # These funcs are different then the Jupyter notebook ones, since I updated env and libs
    img = load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(_RESNET_MODEL.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    """Evaluates if dog is located in the image with provided path.

    Args:
        img_path (str): Path to the image.

    Returns:
        bool: Returns true if dog is detected inside the image
    """
    prediction = ResNet50_predict_labels(img_path)
    # The dog labels are located in [151, 268] region
    return ((prediction <= 268) & (prediction >= 151)) 


if __name__ == '__main__':
    
    # This is the testing feature
    import random
    random.seed(42*42)

    # load filenames in shuffled human dataset
    # human_files = np.array(glob("./data/lfw/*/*"))
    random.shuffle(HUMAN_FILES)

    # print statistics about the dataset
    print('There are %d total human images.' % len(HUMAN_FILES))

    # face_detector(os.fspath(HUMAN_FILES[20]), show_faces=True)
    face_detector(os.fspath('/home/nikola/Pictures/Webcam/tikic.jpg'), plot_faces=True)
    _dog_detected = dog_detector(os.fspath('/home/nikola/Pictures/Webcam/tikic.jpg'))
    print(f"Dog detected in the image: {_dog_detected}")