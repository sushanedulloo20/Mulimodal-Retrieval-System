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
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
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

file_words=open("words.pkl","rb")
words=pickle.load(file_words)
file_image=open("image_features.pkl","rb")
image_list=pickle.load(file_image)
image_removal_data=pd.read_csv("after_image_removal.csv")
preprocessed_data=pd.read_csv("preprocessed.csv")
actual_data=pd.read_csv("A2_Data.csv")
idf_values=pickle.load(open("idf_values.pkl","rb"))
tfidf_values=pickle.load(open("tfidf.pkl","rb"))

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
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features

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
def max_similarity(feature,image_features=image_list):
    v1 = feature.flatten()
    
    scores=[]
    i=0
    for f in image_features:
        f1=f[1]
        
        v2=f1.flatten()
        
        sim=cosine_similarity(v1,v2)
        scores.append((sim,f[0]))
        i+=1
    scores=sorted(scores,reverse=True)
    return scores
def Query_review_preprocessing(query:str):
    query=query.lower()
    tokenized_query=tokenizer.tokenize(query)
    t1=[word for word in tokenized_query if word[0] not in punct]
    t2=[word for word in t1 if word not in new_sp]
    t3=[lem.lemmatize(word,pos='v') for word in t2]
    t4=[stemmer.stem(word) for word in t3]

    new_query=" ".join(t4)
    return new_query
def Calcute_tfidf_query(query:str):
    tokenized_query=query.split()
    vector={}
    for word in words:
        tf=tokenized_query.count(word)
        idf=idf_values[word]
        tfidf=tf*idf
        vector[word]=tfidf
    return vector
    

input_image=input("Give image url: ")
review_input=input("Give review as input: ")
review_input=Query_review_preprocessing(review_input)

vector_review_input=Calcute_tfidf_query(review_input)

features_image_input=Feature_extraction_images(input_image)
features_image_input=features_image_input/np.linalg.norm(features_image_input)

scores=max_similarity(features_image_input)
results=[]
index_stored=[]
i=0
while len(results)!=3:
    if scores[i][1] not in index_stored:
        results.append(scores[i])
        index_stored.append(scores[i][1])
    i+=1
n=1
comp_scores=[]
for result in results:
    id=result[1]
    cosine_image_simlarity=result[0]
    print(n)
    urls=image_removal_data["Image"][id]
    review=image_removal_data["Review Text"][id]
    print(f"Image url: {urls}")
    print(f"Review:{review}")
    print(f"Cosine similarity images:{result[0]}")
    v1=formvectors(vector_review_input)
    v2=formvectors(tfidf_values[id])
    cosine=cosine_similarity(v1,v2)
    print(f"Cosine simalrity of review:{cosine}")
    composite_score=((cosine+result[0])/2)
    print(f"Composite score: {composite_score}")
    comp_scores.append((composite_score,id,"I"))
    n=n+1
pickle.dump(comp_scores,open("composite_score_imagebased.pkl","wb"))





