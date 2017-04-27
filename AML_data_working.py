
import re
import nltk
import nltk.data
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pickle
import sys 
import os

stop_words = set(stopwords.words('english'))

def count_title(s, title):
    count = 0
    for word in s:
        if word in title:
            count += 1
    return count

def count_important_words(s, top):
    count = 0
    for word in s:
        if word in top:
            count += 1
    return count


def create_training_data(file_name):

    sentences_arr = []
    para_arr = []

    paragraphs = file_name.splitlines()


    for paragraph in paragraphs:
        sentences_arr = []
        para = nltk.sent_tokenize(paragraph)
        if para != []:
            for sentences in para:
                sentences_arr.append(sentences.split())
        if sentences_arr != []:
            para_arr.append(sentences_arr)

    stem_txt_final=[]
    porter_stemmer = PorterStemmer()

    for p in para_arr:
        stem_txt = []
        for s in p:
            temp=[]
            for word in s:
                if word not in stop_words:
                    word=re.sub(r'[^\w]','', word)
                    temp.append(porter_stemmer.stem(word.lower()))
            stem_txt.append(temp)
        stem_txt_final.append(stem_txt)


    title_list = stem_txt_final[0][0][0]


    words = []
    for para in stem_txt_final:
        article_string = ['  '.join(mylist) for mylist in para]
        words.extend(article_string)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(words)
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    top_n = 10
    top_features = [features[i] for i in indices[:top_n]]

    count = 0
    features = []
    for para in stem_txt_final:
        position = 0
        for s in para:
            position += 1
            count += 1
            feat = [position, count, count_title(s, title_list), count_important_words(s, top_features), len(s)]
            features.append(feat)
    return features


d = "/Users/shuyang/Downloads/training/docs/"
features = []
import io
def run():
	count = 0
	for file in os.listdir("/Users/shuyang/Downloads/training/docs"):
    		if file.endswith(".txt"):
    			
        		print(os.path.join("", file))
        		name = d + os.path.join("", file)
        		print name
        		with io.open(name, encoding='ISO-8859-1') as f:
            			read_data1 = f.read()
                        try:
                            count += 1
                            print count
                            read_data = read_data1.decode("utf8")
                            feature = create_training_data(read_data)
                            features.append(feature)
                        except:
                            pass
                        
            			
run()
out_file = open('test_data.txt', 'wb')
pickle.dump(features, out_file)
out_file.close()

# my_list = []
# import pickle
# with open('train_data.txt', 'rb') as in_file:
#    my_list = pickle.load(in_file)


# count = 0
# for article in my_list:
#     count += 1
#     print count
#     print len(article)
