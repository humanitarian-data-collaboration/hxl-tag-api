
# coding: utf-8

# In[ ]:


from hdx.utilities.easy_logging import setup_logging
from hdx.hdx_configuration import Configuration
from hdx.data.dataset import Dataset
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re


# In[ ]:


setup_logging()
Configuration.create(hdx_site='prod', user_agent='A_Quick_Example', hdx_read_only=True)


# In[ ]:


#Check if the dataset has at least 1 resource of the required file type(s).

def check_type(dataset, file_types=[]):
    temp_dataset = Dataset.read_from_hdx(dataset)
    temp_dataset.separate_resources()
    if (len(temp_dataset.resources) > 0):
        if (len(file_types) > 0):
            if (not set(temp_dataset.get_filetypes()).isdisjoint(file_types)): 
                    return True
        else :
            return True
    return False


# In[ ]:


# Check if the dataset is tagged with HXL tag, not provided by HXL

def check_organization(dataset):
    if dataset.get_organization()['title'] != 'Humanitarian Exchange Language(HXL)':
            return True
    return False


# In[ ]:


nltk.download('stopwords')

from nltk.corpus import stopwords

#CLEANING AND GENERATING N-GRAMS

def lower_cols(lst):
    #convert data to lowercases
    #QUESTION: will I miss anyt important information? 
    return [word.lower() for word in lst]

#Question: is HXL Core Schema.csv something we can use for comparing words??
#This method is going to take up a lot of space and time. Is it worth it? Are there any other ways to go about it? 

def remove_chars(lst):
    #remove punctuation characters such as ",", "(", ")", """, ":", "/", and "."
    #NOTE: PRESERVES WHITE SPACE.
    #QUESTION: any other characters we should be aware of? Is this a good idea? I'm inspecting each word individually.
    #Any potential pitfalls? 
    cleaned = [re.sub('\s+', ' ', mystring).strip() for mystring in lst]
    cleaned = [re.sub(r'[[^A-Za-z0-9\s]+]', ' ', mystr) for mystr in cleaned]
    cleaned = [mystr.replace('_', ' ') for mystr in cleaned]
    return cleaned

stopWords = set(stopwords.words('english'))

def remove_stop_words(data_lst):
    #remove stopwords from the data including 'the', 'and' etc. 
    wordsFiltered = []
    for w in data_lst:
        if w not in stopWords:
            wordsFiltered.append(w)
    return wordsFiltered

def clean_cols(data):
    data = lower_cols(data)
    data = remove_chars(data)
    return data


# In[ ]:


# Download one dataset with certain type(s), read it into Dataframe, 
# add all headers, tags and dataset names to our DataFrame,
# and delete the dataset

def process_dataset_2(dataset, file_type, dataframe, download_path, index, row_limit = 10):
    global count
    organization = ""
# Download one dataset and read it into a DataFrame 
    if (file_type == None):
        url, path = dataset.resources[0].download(download_path)
        pandas_dataset = pd.read_csv(path)
    else:
        if (file_type not in dataset.get_filetypes()):
            return 'Error: Required file type not in dataset OR dataset does not contain any resources.'
        try:
            url, path = dataset.resources[dataset.get_filetypes().index(file_type)].download(download_path)
            organization = dataset.get_organization()['title']
            print('Resource URL %s downloaded to %s' % (url, path))
            pandas_dataset = pd.read_csv(path, encoding='latin-1')
            pandas_dataset = pandas_dataset.head(row_limit)
        except:
            return 'Unknown error.'
     
    #if "HXL" in os.path.basename(path) or "hxl" in os.path.basename(path):
        #return dataset_df
    
# Add headers, tags and data to our DataFrame if current dataset not empty
        if (not pandas_dataset.empty):
            dataset_df = pandas_dataset
            headers = list(dataset_df.columns.values)
            headers = clean_cols(headers)
            tags = list(dataset_df.iloc[0,:])
            for i in range(len(headers)):
                try:
                    splitted = re.split('[(^\s+)+#]', tags[i])
                    splitted = list(filter(None, splitted))
                    hashtag = splitted[0]
                    attributes = splitted[1:]
                    dic = {'Header': headers[i], 'Tag': hashtag, 'Attributes': attributes, 
                           'Data': list(dataset_df.iloc[1:, i]), 
                           'Relative Column Position': (i+1) / len(dataset_df.columns), 
                           'Dataset_name': os.path.basename(path), 
                           'Organization': organization,
                           'Index': index}
                    dataframe.loc[len(dataframe)] = dic
                except:
                    print("Error: different number of headers and tags")
            count += 1
        os.remove(path)
        print("File Removed!")
        return


# In[ ]:


# Search for all datasets with HXL tags

datasets_HXL = Dataset.search_in_hdx('HXL')
len(datasets_HXL)


# In[ ]:


# Create a DataFrame for all headers and tags

col_names = ['Header', 'Tag', 'Attributes','Data','Relative Column Position','Dataset_name', 'Organization','Index']
headers_and_tags= pd.DataFrame(columns = col_names)


# In[ ]:


count = 0
for i in range(150):
    rand_dataset = np.random.randint(0, len(datasets_HXL))
    process_dataset_2(datasets_HXL[rand_dataset], 'CSV', headers_and_tags, './datasets', count)
    print(i)


# In[ ]:


headers_and_tags.to_excel("headerandtag.xlsx")


# In[ ]:


headers_and_tags.head(200)


# In[ ]:


#counting column names that appear the most frequently in the set:
from collections import Counter
counts = Counter(headers_and_tags.iloc[:, 0])
print(counts)


# In[ ]:


#implementing Bag Of Words Model
#STEPS
#1) collect all col_names that have the same underlying meaning ex: deaths/fatalities etc. (see tags) 
#2) construct another dataframe that consists of the following headers: 
#3) [Header, Word Frequency]... could be a dictionary or multi-indexed dataframe where second column is all the words that 
#appear in the data under the given col name and the value would be count of word that appears. 
#4) --> feature. 


# In[ ]:


#implementing n-grams Model
import nltk
    
def generate_n_grams(data_lst, n):
    cleaned = remove_chars(list(data_lst))
    cleaned = clean_cols(cleaned)
    cleaned = remove_stop_words(cleaned)
    #make sure that n_grams 'refresh' when a new dataset is encountered!!!!   
    return list(ngrams(cleaned, n))


# In[ ]:


#creating a n-gram frequency table (this is hopefully useful for determining if adjacent columns have effect on tags)
#this can also be applied to exploring correlation between tags and attributes. 
#any applications to the data itself? Should I treat all of the data as a single list of words? Example:
#currently thinking of using BOW to the data itself? 

def count_stats_grams(two_d_arr):
    #np.unique 'axis' attribute doesn't work on my computer... 
    lst = np.array([])
    count = 0
    singles_count = 0
    multiples_count = 0
    for arr in two_d_arr:
        if arr not in lst:
            count += 1
            np.append(lst, arr)
        if two_d_arr.count(arr) == 1:
            singles_count += 1
        if two_d_arr.count(arr) > 1:
            multiples_count += 1
    check = count - singles_count
    assert(check == multiples_count)
    return count, singles_count, multiples_count

def n_gram_freqs(dataframe, max_n = 4):
    n_gram_cols = ['n-gram', 'data' ,'unique ngrams', 'multiples', 'singles']
    n_gram_freqs = pd.DataFrame(columns = n_gram_cols)
    for i in range(max_n):
        n = i+1
        n_grams = generate_n_grams(dataframe['Header'], n)
        unique_n_grams, singles, multiples = count_stats_grams(n_grams)
        row = {'n-gram': n, 
              'data': n_grams,
              'unique ngrams': unique_n_grams,
              'multiples': multiples,
              'singles': singles}
        n_gram_freqs.loc[len(n_gram_freqs)] = row
    return pd.DataFrame(n_gram_freqs)  


# In[ ]:


#Takes a data row and cleans it for model input
import ast
import itertools 

def word_extract(row):
    ignore = ['nan']
    #words = ast.literal_eval(row)
    no_white = [i.lstrip() for i in row if i not in ignore and not isinstance(i, float)]
    #divide_space = [i.split() for i in no_white]
    cleaned_text = [w.lower() for w in no_white if w not in ignore]
    return cleaned_text

long_string = []
for i in headers_and_tags['Data']:
    result_by_tag = word_extract(i)
    holder_list = ' '.join(result_by_tag)
    long_string.append(holder_list)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

corpus = long_string
X_vecs = vectorizer.fit_transform(corpus)
#np.set_printoptions(threshold = np.inf)
#print(X.toarray())


# In[ ]:


print(np.shape(X_vecs.toarray()))


# In[ ]:


#testing MLP Classifier
from sklearn.neural_network import MLPClassifier
from fastText import load_model
from sklearn.model_selection import train_test_split

fasttext_model = 'wiki.en.bin'
fmodel = load_model(fasttext_model)
print("Pre-trained model loaded successfully!\n")


# In[ ]:

#Classification using only headers
df = headers_and_tags
df['Header_embedding'] = df['Header'].map(lambda x: fmodel.get_sentence_vector(str(x)))
df['Organization_embedded'] = df['Organization'].map(lambda x: fmodel.get_sentence_vector(str(x)))
#df['data_embedding'] = df['Data'].map(lambda lst: [fmodel.get_sentence_vector(str(x)) for x in lst])
#df['ngram']
print("Word embeddings extracted!\n")

#test = df['Header_embedding'] + df['Organization_embedded']
#combining so that the matrix will be in the right shape...
#not sure if this will make sense... 
#print(np.shape(test))

X_train, X_test, y_train, y_test = train_test_split(df['Header_embedding'], 
                                                    df['Tag'], test_size=0.33, random_state=0)

clf = MLPClassifier(activation='relu', alpha=0.001, epsilon=1e-08, hidden_layer_sizes=150, solver='adam')

#have to ensure X_train is the right shape
#temp = X_train.to_csv('temp.csv', header = False)
#X_train = pd.read_csv('temp.csv', header = None)
#os.remove('temp.csv')
#X_train.index = X_train['0']
#X_train.columns = [x for x in range(len(X_train.columns))]
#X_train = np.reshape(X_train, (len(X_train), len(X_train.columns)))
clf.fit(X_train.values.tolist(), y_train.values.tolist())
test_score = clf.score(X_test.tolist(), y_test.tolist())
print("Classification accuracy on test set: %s" %test_score)


# In[ ]:


#classification using organization
df = headers_and_tags
df['Header_embedding'] = df['Header'].map(lambda x: fmodel.get_sentence_vector(str(x)))
df['Organization_embedded'] = df['Organization'].map(lambda x: fmodel.get_sentence_vector(str(x)))
#df['data_embedding'] = df['Data'].map(lambda lst: [fmodel.get_sentence_vector(str(x)) for x in lst])
#df['ngram']
print("Word embeddings extracted!\n")

#test = df['Header_embedding'] + df['Organization_embedded']
#combining so that the matrix will be in the right shape...
#not sure if this will make sense... 
#print(np.shape(test))

X_train, X_test, y_train, y_test = train_test_split(df['Organization_embedded'], 
                                                    df['Tag'], test_size=0.33, random_state=0)

clf = MLPClassifier(activation='relu', alpha=0.001, epsilon=1e-08, hidden_layer_sizes=150, solver='adam')

#have to ensure X_train is the right shape
#temp = X_train.to_csv('temp.csv', header = False)
#X_train = pd.read_csv('temp.csv', header = None)
#os.remove('temp.csv')
#X_train.index = X_train['0']
#X_train.columns = [x for x in range(len(X_train.columns))]
#X_train = np.reshape(X_train, (len(X_train), len(X_train.columns)))
clf.fit(X_train.values.tolist(), y_train.values.tolist())
test_score = clf.score(X_test.tolist(), y_test.tolist())
print("Classification accuracy on test set: %s" %test_score)


# In[ ]:


from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
ngrams = generate_n_grams(headers_and_tags['Header'], 3)
vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
X_vec_grams = vectorizer.fit_transform(ngrams)
print(np.shape(X_vec_grams.toarray()))
print(np.shape(X_vecs.toarray()))


# In[ ]:


#testing MLP Classifier on BOW 
df_2 = headers_and_tags
df_2['BOW_counts'] = X_vecs.toarray()

X_train, X_test, y_train, y_test = train_test_split(X_vecs.toarray(), 
                                                    df['Tag'], test_size=0.33, random_state=0)

clf = MLPClassifier(activation='relu', alpha=0.001, epsilon=1e-08, hidden_layer_sizes=150, solver='adam')

#have to ensure X_train is the right shape
#temp = X_train.to_csv('temp.csv', header = False)
#X_train = pd.read_csv('temp.csv', header = None)
#os.remove('temp.csv')
#X_train.index = X_train['0']
#X_train.columns = [x for x in range(len(X_train.columns))]
#X_train = np.reshape(X_train, (len(X_train), len(X_train.columns)))
clf.fit(X_train, y_train)
test_score = clf.score(X_test, y_test)
print("Classification accuracy on test set: %s" %test_score)


# In[ ]:


#testing MLP Classifier on ngrams
df_3 = headers_and_tags

X_train, X_test, y_train, y_test = train_test_split(X_vec_grams.toarray(), 
                                                    df['Tag'][0:len(X_vec_grams.toarray())], test_size=0.33, random_state=0)

clf = MLPClassifier(activation='relu', alpha=0.001, epsilon=1e-08, hidden_layer_sizes=150, solver='adam')

#have to ensure X_train is the right shape
#temp = X_train.to_csv('temp.csv', header = False)
#X_train = pd.read_csv('temp.csv', header = None)
#os.remove('temp.csv')
#X_train.index = X_train['0']
#X_train.columns = [x for x in range(len(X_train.columns))]
#X_train = np.reshape(X_train, (len(X_train), len(X_train.columns)))
clf.fit(X_train, y_train)
test_score = clf.score(X_test, y_test)
print("Classification accuracy on test set: %s" %test_score)


# In[ ]:


#transforming data into right form

df_target = headers_and_tags
df_target['BOW_counts'] = [item for item in X_vecs.toarray()]
n = len(X_vec_grams.toarray())
df_target = df.iloc[0:n, :]
df_target['ngrams_counts'] = [item for item in X_vec_grams.toarray()]
df_target = df_target[['Header_embedding', 
                      'Organization_embedded',
                      'BOW_counts',
                      'ngrams_counts']]
df_target.head()
#print(X_vecs.toarray()[0])


# In[ ]:


#Using VotingClassifier and Pipeline to combine all features to predict the 'most voted' output 

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator

# custom transformer for sklearn pipeline
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self

X_train, X_test, y_train, y_test = train_test_split(df_target, df['Tag']
                                                   [0:len(X_vec_grams.toarray())])

header_pipe = Pipeline([
        ('col_extract', ColumnExtractor(cols = 1)), 
        ('clf', clf)
    ])

org_pipe = Pipeline([
        ('col_extract', ColumnExtractor(cols = 2)), 
        ('clf', clf)
    ])

BOW_pipe = Pipeline([
        ('col_extract', ColumnExtractor(cols = 3)), 
        ('clf', clf)
    ])

Ngram_pipe = Pipeline([
        ('col_extract', ColumnExtractor(cols = 4)), 
        ('clf', clf)
    ])

eclf = VotingClassifier(estimators = [('p1', header_pipe),
                                     ('p2', org_pipe),
                                     ('p3', BOW_pipe),
                                     ('p4', Ngram_pipe)], 
                                   voting = 'soft', 
                                   weights = [1, 0.5, 0.5, 0.5])
eclf.fit(X_train, y_train)
eclf_score = eclf.score(X_test, y_test)
print("Classification accuracy on test set: %s" %eclf_score)                            

