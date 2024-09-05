import pickle
import numpy as np
import pandas as pd
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
import string
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import requests
from io import BytesIO
from PIL import Image
from nltk.stem.porter import PorterStemmer 
import tensorflow as tf
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

data=pd.read_csv("preprocessed.csv")
df=pd.read_csv("after_image_removal.csv")
file_words=open("words.pkl","rb")
file_idf=open("idf_values.pkl",'rb')
file_tfidf=open("tfidf.pkl","rb")
tfidf_dic=pickle.load(file_tfidf)
idf_values=pickle.load(file_idf)
image_features=pickle.load(open("image_features.pkl","rb"))
words=pickle.load(file_words)
stemmer = PorterStemmer()
lem=WordNetLemmatizer()
sp=set(stopwords.words("english"))
new_sp=[]

tokenizer=WordPunctTokenizer()
punct=string.punctuation
for word in sp:
    tokenized=tokenizer.tokenize(word)
    for i in tokenized:
        if i not in punct:
            new_sp.append(i)
new_sp=set(new_sp)
def preprocess_image(image):
    # Randomly flip the image horizontally
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)

    # Adjust brightness and contrast randomly
    alpha = 1.0 + np.random.uniform(-0.5, 0.5)  # Random alpha value for brightness adjustment
    beta = 0.0  # Constant beta value for contrast adjustment
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted_image
def adjust_brightness_contrast(image, target_brightness=0, target_contrast=0):
   
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

def preprocessing(corpus:str):
    corpus=corpus.lower()
    tokenized_corpus=tokenizer.tokenize(corpus)
    t1=[word for word in tokenized_corpus if word[0] not in punct]
    t2=[word for word in t1 if word not in new_sp]
    t3=[lem.lemmatize(word,pos='v') for word in t2]
    t4=[stemmer.stem(word) for word in t3]

    new_query=" ".join(t4)
    return new_query

def cosine_similarity(vector1, vector2):
    
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1) 
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity
def formvectors(dict1:dict,words=words):
    v1=[]
    
    for w in words:
        v1.append(dict1[w])
        
    return v1
def maxsimilarity(v:list,data=data,words=words):
    v=np.array(v)
    scores=[]
    for i in range(0,994):
        dicti=tfidf_dic[i]
        vector=np.array(formvectors(dicti))
        sim=cosine_similarity(v,vector)
        scores.append((sim,i))
    scores_updated =sorted(scores,reverse=True)
    
    return scores_updated
def Convert_query(q:str,idf_v=idf_values,words=words):
    q_new=preprocessing(q)
    q_new_list=q_new.split()
    vector=[]
    for word in words:
        tf=0
        idf=0

        if word in q_new_list:
            tf=(q_new_list.count(word))/len(q_new_list)
            idf=idf_v[word]
        tfidf=tf*idf
        vector.append(tfidf)
    return vector
def average(numbers:list):
    sum=0
    for score in numbers:
        sum+=score
    avg=(sum/len(numbers))
    return avg

    
        
            

image_input=input("Enter Image url: ")
query=input("Enter your review: ")
features_image_input=Feature_extraction_images(image_input)
features_image_input=features_image_input/np.linalg.norm(features_image_input)
v=Convert_query(query)
scores=maxsimilarity(v)
comp_scores=[]
for i in range(3):
    id=scores[i][1]
    urls=df["Image"][id]
    print(i+1)
    print(f"Image urls: {urls}")
    review=df["Review Text"][id]
    print(f"Review:{review}")
    scores_cosine=[]
    for features in image_features:
        if id==features[0]:
            v1=features_image_input.flatten()
            v2=features[1].flatten()
            scores_cosine.append(cosine_similarity(v1,v2))
    avg=average(scores_cosine)
    
    print(f"cosine simalrity of image:{avg}")
    print(f"cosine similarity of review:{scores[i][0]}")
    composite_score=((avg+scores[i][0])/2)
    comp_scores.append((composite_score,id,"T"))
    print(f"composite score: {composite_score}")
pickle.dump(comp_scores,open("composite_scores_textbased.pkl","wb"))
    








        
        
    
