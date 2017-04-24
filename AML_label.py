import nltk
from rouge import Rouge 
import pickle
import os
#import sys
#import io
#nltk.download()
#from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
#path='/Users/shuyang/Downloads/training/docs/alaska.rescue2009.doc.txt'
#path_hlight='/Users/shuyang/Downloads/training/hlights/alaska.rescue2009.hlights.txt'
#path='/Users/shuyang/Documents/Advanced machine learning/Project/bbc/business/003.txt'
#sentences_arr = []
#para_arr = []
#with open(path) as f:
#    read_data1 = f.read()
#read_data = read_data1.decode("utf8")
#
##paragraphs = read_data.split('\n')
#paragraphs = [s.strip() for s in read_data.splitlines()]
#
#
#for paragraph in paragraphs:
#    sentences_arr = []
#    para = nltk.sent_tokenize(paragraph)
#    if para != []:
#        for sentences in para:
#                sentences_arr.append(sentences.split())
#    if sentences_arr != []:
#        para_arr.append(sentences_arr)
#
#stem_txt_final=[]
#stem_txt=[]
#stop_words = set(stopwords.words('english'))
#porter_stemmer = PorterStemmer()
#
#for para in para_arr:
#    stem_txt=[]
#    for s in para:
#        temp=[]
#        for word in s:
#            word=re.sub(r'[^\w]','', word)
#            word=word.lower()
#            if word not in stop_words:
#                temp.append(porter_stemmer.stem(word))
#        stem_txt.append(temp)
#    stem_txt_final.append(stem_txt)
#
#
#words = []
#vectorizer = TfidfVectorizer()
#for para in stem_txt_final:
#    article_string = ['  '.join(mylist) for mylist in para]
#    words.extend(article_string)
#X = vectorizer.fit_transform(words)
#indices = np.argsort(vectorizer.idf_)[::-1]
#features = vectorizer.get_feature_names()
#top_n = 10
#top_features = [features[i] for i in indices[:top_n]]
#print top_features


#rouge = Rouge()
#with open(path) as f:
#    docs = f.read()
#with open(path_hlight) as f:
#    hlights = f.read()
#
#paragraphs = docs.split('\n')
#
#label_list=[]
#for paragraph in paragraphs:
#    if paragraph != '':
#        label_list.append(paragraph)
#
#score_result=[]
#for sentence_label in label_list:
#    
#    scores = rouge._get_avg_scores(sentence_label, hlights)
#    scores=scores['rouge-1']['f']
#    score_result.append(scores)
#
#index_sorted=sorted(range(len(score_result)), key=lambda i:score_result[i])
#index_sorted.reverse()
#top_4=index_sorted[:4]
#label=np.array(score_result)
#label[top_4]=1
#other=index_sorted[4:]
#label[other]=0
#label=list(label)
#label=[int(i) for i in label]
rouge = Rouge()
d='/Users/shuyang/Downloads/test/docs/'
h='/Users/shuyang/Downloads/test/hlights/'
#d='/Users/shuyang/Downloads/traning/docs/'
#h='/Users/shuyang/Downloads/training/hlights/'

result=[]
#s = '/Users/shuyang/Downloads/test/{0}/'.format(doc_type)
#for file in os.listdir("/Users/shuyang/Downloads/training/docs"):
for file in os.listdir("/Users/shuyang/Downloads/test/docs"):
    if file.endswith(".txt"):
        os_path=os.path.join("", file)
        os_path=os_path[:-7]
        print os_path
        name_docs= d + os.path.join("", file)
        name_hlights=h+os_path+'hlights.txt'
        print name_docs
        with open(name_docs) as f:
            docs = f.read()
        with open(name_hlights) as f:
            hlights = f.read()
        paragraphs = docs.split('\n')
        label_list=[]
        for paragraph in paragraphs:
            if paragraph != '':
                label_list.append(paragraph)
        
        score_result=[]
        for sentence_label in label_list:
            try:
                scores = rouge._get_avg_scores(sentence_label, hlights)
            except:
                pass
            #print scores
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
        result.append(label)
        #f.close()


out_file = open('train_label.txt', 'wb')
pickle.dump(result, out_file)
out_file.close()
#with open('test_label.txt', 'rb') as f:
#    my_list = pickle.load(f)
#print my_list

#with open('train_label.txt', 'rb') as f:
#    my_list = pickle.load(f)
#print my_list




        
