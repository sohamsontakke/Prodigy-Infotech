import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py

st.header("Food Recognition(using InceptionV3)")

#Function to allow us to upload, and display the image uploaded
def main():
    file_uploaded = st.file_uploader("Choose file to upload", type = ['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)
        
#Function that uses the model specified, to make predictions.
def predict_class(image):
    classifier_model = tf.keras.models.load_model(r'/content/gdrive/MyDrive/InceptionModel/inception_model.hdf5')
    shape = ((299,299,3))
    #Use hub to load the model
    model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])
    test_image = image.resize((299,299))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    target_names = [ 'rice', 
                    'tempura bowl',
                    'bibimbap',
                    'toast',
                    'croissant',
                    'roll bread',
                    'rasin bread',
                    'chip butty',
                    'hamburger',
                    'pizza',
                    'sandwiches',
                    'eels on rice',
                    'udon noodle',
                    'tempura udon',
                    'soba noodle',
                    'ramen noodle',
                    'beef noodle',
                    'tensin noodle',
                    'fried noodle',
                    'spaghetti',
                    'Japanese-style pancake',
                    'takoyako',
                    'pilaf',
                    'gratin',
                    'sauteed vegetables',
                    'croquette',
                    'grilled eggplant',
                    'sauteed spinach',
                    'vegetable tempura',
                    'miso soup',
                    'potage',
                    'sausage',
                    'oden',
                    'chicken and egg on rice',
                    'omelet',
                    'ganmodoki',
                    'jiaozi',
                    'stew',
                    'teriyaki grilled fish',
                    'fried fish',
                    'grilled salmon',
                    'salmon meuniere',
                    'sashimi',
                    'grilled pacific saury',
                    'pork cutlet on rice',
                    'sukiyaki',
                    'sweet and sour pork',
                    'lightly roasted fish',
                    'steamed egg hotchpotch',
                    'tempura',
                    'fried chicken',
                    'sirloin cutlet',
                    'nanbanzuke',
                    'boiled fish',
                    'seasoned beef with potatoes',
                    'beef curry',
                    'hambarg steak',
                    'steak',
                    'dried fish',
                    'ginger pork saute',
                    'spicy chili-flavoured tofu',
                    'yakitori',
                    'cabbage roll',
                    'omelet',
                    'egg sunny-side up',
                    'natto',
                    'sushi',
                    'cold tofu',
                    'egg roll',
                    'chilled noodle',
                    'stir-fried beef and peppers',
                    'simmered pork',
                    'boiled chicken and vegetables',
                    'sashimi bowl',
                    'sushi bowl',
                    'fish-shaped pancake with bean jam',
                    'shrimp with chilli sauce',
                    'chicken rice',
                    'roast chicken',
                    'steamed meat dumpling',
                    'omelet with fried rice',
                    'cutlet curry',
                    'spaghetti meat sauce',
                    'fried shrimp',
                    'fried rice']
    #Get predictions from model                
    predictions = model.predict(test_image)
    #Softmax activation will takes a real vector as input and convert it in to a vector of categorical probabilities. 
    values = tf.nn.softmax(predictions[0])
    values = values.numpy()
    image_class = target_names[np.argmax(values)]
    result = "The image uploaded is: {}".format(image_class)
    return result
    
if __name__ == "__main__":
    main()