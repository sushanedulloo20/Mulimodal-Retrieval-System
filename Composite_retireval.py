import pickle
import pandas as pd
text_ranking=pickle.load(open("composite_scores_textbased.pkl","rb"))

image_ranking=pickle.load(open("composite_score_imagebased.pkl","rb"))

data=pd.read_csv("after_image_removal.csv")

ranking=text_ranking+image_ranking

ranking=sorted(ranking,key=lambda x: x[0],reverse=True)

final_ranking=[]
id_teaken=[]
i=0
while len(final_ranking)!=3:
    if ranking[i][1] not in id_teaken:
        final_ranking.append(ranking[i])
        id_teaken.append(ranking[i][1])
    i=i+1


for i in range(3):
    result=final_ranking[i]
    id=result[1]
    if result[2]=="T":
        composite_score=result[0]
        image_urls=data["Image"][id]
        review=data["Review Text"][id]
        print(f"{i+1}")
        print(f"Image urls: {image_urls}")
        print(f"Review: {review}")
        print(f"Composite score: {composite_score}")
    if result[2]=="I":
        composite_score=result[0]
        image_urls=data["Image"][id]
        review=data["Review Text"][id]
        print(f"{i+1}")
        print(f"Image urls: {image_urls}")
        print(f"Review: {review}")
        print(f"Composite score: {composite_score}")


