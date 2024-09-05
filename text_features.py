import math
import pandas as pd
import pickle
data=pd.read_csv("preprocessed.csv")

def calculate_tf(word:str,document:list):
    
    tf=(document.count(word))/len(document)
    return tf
def calculate_idf(documents:list):
    idf_values={}
    words=[]
    i=0
    for sentence in documents:

        words_list_doc=str(sentence).split()
        for w in words_list_doc:
            if w not in words:
                words.append(w)
        i+=1
    print(sentence)
    tokenized_doc=[str(sentence).split() for sentence in documents]
    for word in words:
        c=0
        for t in tokenized_doc:
            if word in t:
                c+=1
        idf_w=math.log(len(documents)/(c))
        idf_values[word]=idf_w
    return words,idf_values
def convert_tfidf_(documents:list):
    words,idf_values=calculate_idf(documents)
    tfidf_vectors=[]
    for doc in documents:
        l=str(doc).split()
        tfidf_doc={}
        for word in words:
            tf=calculate_tf(word,l)
            idf=idf_values[word]
            tfidf=tf*idf
            tfidf_doc[word]=tfidf
        tfidf_vectors.append(tfidf_doc)
    return tfidf_vectors,words


reviews=data['Review Text'].to_list()
print(type(reviews[0]))
words,idf_values=calculate_idf(reviews)
tf_idf_vectors,y=convert_tfidf_(reviews)
print(words)
print(idf_values)
print(tf_idf_vectors)

file_tfidf=open("tfidf.pkl","wb")
file_idf=open("idf_values.pkl","wb")
file_words=open("words.pkl","wb")
pickle.dump(idf_values,file_idf)
pickle.dump(words,file_words)
pickle.dump(tf_idf_vectors,file_tfidf)


        






    
