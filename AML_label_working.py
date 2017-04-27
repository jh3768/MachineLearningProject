import nltk
from rouge import Rouge 
import pickle
import os
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
rouge = Rouge()
d='/Users/shuyang/Downloads/training/docs/'
h='/Users/shuyang/Downloads/training/hlights/'
name_del=['allen.pope.doc.txt','behind.battlefield.doc.txt','best.gangsters.doc.txt','btravel.overview.doc.txt','btsc.first.lady.travel2.doc.txt','btsc.lemon.doc.txt','cb.lies.on.resumes.doc.txt','commentary.johnson.doc.txt']
result=[]
name=[]
for file in os.listdir("/Users/shuyang/Downloads/training/docs"):
#for file in os.listdir("/Users/shuyang/Downloads/test/docs"):
    if file.endswith(".txt"):
        if os.path.join("", file) not in name_del:
            os_path=os.path.join("", file)
            os_path=os_path[:-7]
            name_docs= d + os.path.join("", file)
            name_hlights=h+os_path+'hlights.txt'
            print name_docs
            name.append(name_docs)
            with open(name_docs) as f:
                docs = f.read()
            with open(name_hlights) as f:
                hlights = f.read()
            paragraphs = docs.split('\n')
            label_list=[]
            for paragraph in paragraphs:
                para = nltk.sent_tokenize(paragraph)

                if para != []:
                    for sentences in para:
                        label_list.append(sentences)
            
            score_result=[]
            for sentence_label in label_list:
                try:
                    scores = rouge._get_avg_scores(sentence_label, hlights)
                except:
                    pass
                try:
                    scores=scores['rouge-1']['f']
                except:
                    scores=0
                score_result.append(scores)
            
            index_sorted=sorted(range(len(score_result)), key=lambda i:score_result[i])
            index_sorted.reverse()
            top_4=index_sorted[:4]
            label=np.array(score_result)
            label[top_4]=1
            other=index_sorted[4:]
            label[other]=0
            label=list(label)
            label=[int(i) for i in label]
            print len(label)
            result.append(label)
        


out_file = open('test_label.txt', 'wb')
pickle.dump(result, out_file)
out_file.close()

#out_file = open('train_label.txt', 'wb')
#pickle.dump(result, out_file)
#out_file.close()
#with open('test_label.txt', 'rb') as f:
#    my_list = pickle.load(f)



        
