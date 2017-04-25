
# coding: utf-8

# In[1]:

#from pyrouge import Rouge155
#from rouge import Rouge
import re
import nltk
#import nltk.data
#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pickle
import sys 
import os


#nltk.download("punkt")
#stop_words = set(stopwords.words('english'))

stop_words = {u'a',
 u'about',
 u'above',
 u'after',
 u'again',
 u'against',
 u'ain',
 u'all',
 u'am',
 u'an',
 u'and',
 u'any',
 u'are',
 u'aren',
 u'as',
 u'at',
 u'be',
 u'because',
 u'been',
 u'before',
 u'being',
 u'below',
 u'between',
 u'both',
 u'but',
 u'by',
 u'can',
 u'couldn',
 u'd',
 u'did',
 u'didn',
 u'do',
 u'does',
 u'doesn',
 u'doing',
 u'don',
 u'down',
 u'during',
 u'each',
 u'few',
 u'for',
 u'from',
 u'further',
 u'had',
 u'hadn',
 u'has',
 u'hasn',
 u'have',
 u'haven',
 u'having',
 u'he',
 u'her',
 u'here',
 u'hers',
 u'herself',
 u'him',
 u'himself',
 u'his',
 u'how',
 u'i',
 u'if',
 u'in',
 u'into',
 u'is',
 u'isn',
 u'it',
 u'its',
 u'itself',
 u'just',
 u'll',
 u'm',
 u'ma',
 u'me',
 u'mightn',
 u'more',
 u'most',
 u'mustn',
 u'my',
 u'myself',
 u'needn',
 u'no',
 u'nor',
 u'not',
 u'now',
 u'o',
 u'of',
 u'off',
 u'on',
 u'once',
 u'only',
 u'or',
 u'other',
 u'our',
 u'ours',
 u'ourselves',
 u'out',
 u'over',
 u'own',
 u're',
 u's',
 u'same',
 u'shan',
 u'she',
 u'should',
 u'shouldn',
 u'so',
 u'some',
 u'such',
 u't',
 u'than',
 u'that',
 u'the',
 u'their',
 u'theirs',
 u'them',
 u'themselves',
 u'then',
 u'there',
 u'these',
 u'they',
 u'this',
 u'those',
 u'through',
 u'to',
 u'too',
 u'under',
 u'until',
 u'up',
 u've',
 u'very',
 u'was',
 u'wasn',
 u'we',
 u'were',
 u'weren',
 u'what',
 u'when',
 u'where',
 u'which',
 u'while',
 u'who',
 u'whom',
 u'why',
 u'will',
 u'with',
 u'won',
 u'wouldn',
 u'y',
 u'you',
 u'your',
 u'yours',
 u'yourself',
 u'yourselves'}



# In[2]:

#rouge = Rouge()
#print dir(rouge)

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



# In[3]:

def create_training_data(file_name):

    sentences_arr = []
    para_arr = []

    paragraphs = file_name.splitlines()

    #print paragraphs

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
    #print top_features

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



# In[10]:

#print sys.getdefaultencoding()


# In[11]:

#import sys
# sys.setdefaultencoding() does not exist, here!
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF-8')


# In[9]:

#print sys.getdefaultencoding()


# In[ ]:



#d = "/Users/cammyluffy/Downloads/4772Project/AML_train/docs/"
d = "/Users/shuyang/Downloads/training/docs/"
features = []
import io
def run():
	count = 0
	#out_file = open('test_X.txt', 'wb')
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
                        
            			#pickle.dump(feature, out_file)                
            			#f.close()

#print features
#print create_training_data(read_data)
#out_file = open('train_X.txt', 'wb')
#pickle.dump(features, out_file)


# In[6]:
run()
#print features
out_file = open('test_data.txt', 'wb')
pickle.dump(features, out_file)
#print len(features[0])
out_file.close()

#print

# In[ ]:
# my_list = []
# import pickle
# with open('train_data.txt', 'rb') as in_file:
#    my_list = pickle.load(in_file)


# count = 0
# for article in my_list:
#     count += 1
#     print count
#     print len(article)

# In[ ]:




# In[ ]:



