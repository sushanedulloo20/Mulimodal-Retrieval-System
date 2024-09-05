import pandas as pd
from nltk.tokenize import WordPunctTokenizer
import string
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import requests
from io import BytesIO
from PIL import Image
data=pd.read_csv("A2_Data.csv")
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
def isImageNotfound(url):
    response = requests.get(url)
    try:
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        img=None
    return img is None
image_list=[]
for i in range(1000):
    print(i)
    img_i=[]
    images=data["Image"][i]
    imageslist=extract_http_links(images)
    x=False
    y=False
    for im in imageslist:
        if isImageNotfound(im):
            if y==False:
                x=True

            pass
        else:
            
            img_i.append(im)
            y=True
            x=False
    if x==True:
        data.drop(i,inplace=True)
    image_list.append(img_i)
series=[pd.Series(lst) for lst in image_list]
data["Image"]=pd.DataFrame({'Image': image_list})
data.to_csv("after_image_removal.csv",index=False)