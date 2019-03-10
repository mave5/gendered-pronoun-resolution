import pandas as pd
import os
import spacy
nlp = spacy.load('en_core_web_sm')

# presets
path2data="../data/"



# load data
path2csv=os.path.join(path2data,"test_stage_1.tsv")
test1_df=pd.read_csv(path2csv,delimiter='\t').rename(columns={'A': 'A_Noun', 'B': 'B_Noun'})
test1_df.head()

print (test1_df.columns)
text=test1_df["Text"][0]
test1_df["Pronoun-offset"][0]
print("length of text: %s" %len(text))

doc = nlp(text)

# find name enteties
for entity in doc.ents:
    print(entity.text, entity.label_)

for text in test1_df["Text"]:
    #print(text)
    print(len(text))

for l in range(len(text)-3):
    if text[l:l+3]=="her":
        print(l)
        
        
