import requests
import pandas as pd
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import pickle
import re
import cv2
data=pd.read_csv("after_image_removal.csv")
def adjust_brightness_contrast(image, target_brightness=0, target_contrast=0):
    """
    Adjusts the brightness and contrast of an image to match target values.
    
    Args:
        image: Input PIL image.
        target_brightness: Target brightness value (0 for no change).
        target_contrast: Target contrast value (0 for no change).
    
    Returns:
        Image with adjusted brightness and contrast.
    """
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Calculate current brightness and contrast
    current_brightness = np.mean(image_array)
    current_contrast = np.std(image_array)
    
    # Calculate adjustment factors
    brightness_factor = target_brightness - current_brightness
    contrast_factor = target_contrast / (current_contrast + 1e-5)
    
    # Apply brightness and contrast adjustment
    adjusted_image = image_array + brightness_factor
    adjusted_image = adjusted_image * contrast_factor
    
    # Clip pixel values to valid range
    adjusted_image = np.clip(adjusted_image, 0, 255)
    
    # Convert numpy array back to PIL image
    adjusted_image = Image.fromarray(adjusted_image.astype(np.uint8))
    
    return adjusted_image
def preprocess_image(image):
    # Randomly flip the image horizontally
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)

    # Adjust brightness and contrast randomly
    alpha = 1.0 + np.random.uniform(-0.5, 0.5)  # Random alpha value for brightness adjustment
    beta = 0.0  # Constant beta value for contrast adjustment
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted_image

def extract_http_links(text):
    """
    Extract all HTTP web links from the given text.
    
    Args:
    text (str): The text containing web links.
    
    Returns:
    list: A list of HTTP web links.
    """
    # Regular expression to match HTTP web links
    pattern = r'https?://\S+.jpg'
    
    # Find all matches in the text
    http_links = re.findall(pattern, text)
    
    return http_links
import numpy as np
def Feature_extraction_images(url):
    model = VGG16(weights='imagenet', include_top=False)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    target_brightness = 100
    target_contrast = 50
    img = adjust_brightness_contrast(img, target_brightness, target_contrast)     

    x = image.img_to_array(img)
    # x=preprocess_image(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features
def isImageNotfound(url):
    response = requests.get(url)
    try:
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        img=None
    return img is None
    
image_list=[]
for i in range(994):
    print(i)
    images=data["Image"][i]
    imageslist=extract_http_links(images)
    for im in imageslist:
        feature=Feature_extraction_images(im)
        feature_new=feature/np.linalg.norm(feature)
        image_list.append((i,feature_new))
    print(images)
file=open("image_features.pkl","wb")
pickle.dump(image_list,file)

   



        
        