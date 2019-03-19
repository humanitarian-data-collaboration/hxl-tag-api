from flask import Flask, redirect, request, render_template, url_for, send_from_directory, make_response
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
from jsonrpc import JSONRPCResponseManager, dispatcher
import os
import hashlib
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import pickle
import re
from sklearn.neural_network import MLPClassifier
from fastText import load_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from flask_cors import CORS

from nltk.corpus import stopwords
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

'''Install fastText by doing:
#git clone https://github.com/facebookresearch/fastText.git
#cd fastText
#pip install .

#'wiki.en.bin' needs to be in the same directory as server.py (can be downloaded from 
#https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip)

'''

fasttext_model = 'wiki.en.bin'
fmodel = load_model(fasttext_model)

UPLOAD_FOLDER = '\datasets'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(app.instance_path)

def lower_cols(lst):
    #convert data to lowercases
    #QUESTION: will I miss any important information? 
    return [word.lower() for word in lst]


def remove_chars(lst):
    #remove punctuation characters such as ",", "(", ")", """, ":", "/", and "."
    #NOTE: PRESERVES WHITE SPACE.
    #QUESTION: any other characters we should be aware of? Is this a good idea? I'm inspecting each word individually.
    #Any potential pitfalls? 
    cleaned = [re.sub('\s+', ' ', mystring).strip() for mystring in lst]
    cleaned = [re.sub(r'[[^A-Za-z0-9\s]+]', ' ', mystr) for mystr in cleaned]
    cleaned = [mystr.replace('_', ' ') for mystr in cleaned]
    return cleaned

def clean_cols(data):
    data = lower_cols(data)
    data = remove_chars(data)
    return data

# def process(pandas_dataset, dataframe):
#     if (not pandas_dataset.empty):
#             dataset_df = pandas_dataset
#             headers = list(dataset_df.columns.values)
#             headers = clean_cols(headers)
#     for i in range(len(headers)):
#         try:
#             dic = {'Header': headers[i], 
#                    'Data': list(dataset_df.iloc[1:, i]), 
#                    'Relative Column Position': (i+1) / len(dataset_df.columns), 
#                    'Dataset_name': os.path.basename(path), 
#                    'Organization': organization,
#                    'Index': index}
#             dataframe.loc[len(dataframe)] = dic
#         except:
#             print("Error: different number of headers and tags")
#     return

def preprocess(pandas_dataset, df_target):
    if (not pandas_dataset.empty):
    	organization = 'HDX'   #Replace if datasets contains organization
    	headers = list(pandas_dataset.columns.values)
    	headers = clean_cols(headers)
    for i in range(len(headers)):
        try:
            dic = {'Header': headers[i], 
                   'Data': list(pandas_dataset.iloc[1:, i]), 
                   'Relative Column Position': (i+1) / len(pandas_dataset.columns), 
                   'Organization': organization,
                   'Index': i}
            df_target.loc[len(df_target)] = dic
        except:
            raise Exception("Error: arguments not matched")

    df_result = transform_vectorizers(df_target)
    return df_result

def transform_vectorizers(df_target):
    cols = ['Header_embedding', 'Organization_embedded', 'BOW_counts', 'ngrams_counts', 'features_combined']
    df = pd.DataFrame(columns = cols)
    long_string = []
    for i in df_target['Data']:
        result_by_tag = word_extract(i)
        holder_list = ''.join(result_by_tag)
        long_string.append(holder_list)
    bag_vectorizer = CountVectorizer()
    corpus = long_string
    X_vecs_bag = bag_vectorizer.fit_transform(corpus)
    df['BOW_counts'] = [item for item in X_vecs_bag.toarray()]
    ngrams = generate_n_grams(df_target['Header'], 3)
    ngrams_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    X_vec_grams = ngrams_vectorizer.fit_transform(ngrams)
    df['ngrams_counts'] = pd.Series([item for item in X_vec_grams.toarray()])
    df['Header_embedding'] = df_target['Header'].astype(str).apply(fmodel.get_sentence_vector)
    df['Organization_embedded'] = df_target['Organization'].astype(str).apply(fmodel.get_sentence_vector)
    cols = ['Header_embedding', 'Organization_embedded', 'BOW_counts', 'ngrams_counts']
    df['features_combined'] = df[cols].values.tolist()
    df['features_combined'] = df['features_combined'].apply(lambda x: np.concatenate(x, axis=None))
    return df

def remove_stop_words(data_lst):
    #remove stopwords from the data including 'the', 'and' etc.
    wordsFiltered = []
    for w in data_lst:
        if w not in stopWords:
            wordsFiltered.append(w)
    return wordsFiltered


def word_extract(row):
    ignore = ['nan']
    no_white = [i.lstrip() for i in row if i not in ignore and not isinstance(i, float)]
    cleaned_text = [w.lower() for w in no_white if w not in ignore]
    return cleaned_text

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

from nltk import ngrams
def generate_n_grams(data_lst, n):
    # cleaned = remove_chars(list(data_lst))
    # cleaned = clean_cols(cleaned)
    cleaned = remove_stop_words(data_lst)
    #make sure that n_grams 'refresh' when a new dataset is encountered!!!!   
    return list(ngrams(cleaned, n))

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # file.save(os.getcwd())
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_dataset = pd.read_csv(file)
                # process the untagged dataset
            processed_dataset = preprocess(input_dataset, 
                pd.DataFrame(columns=['Header','Data','Relative Column Position','Organization','Index']))
            model = pickle.load(open("model.pkl", "rb")) #Model needs be named model.pkl, preferably using version 0.20.3

            output_dataset = pd.DataFrame(data = model.predict(processed_dataset['features_combined']))

            resp = make_response(output_dataset.to_csv())
            resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
            resp.headers["Content-Type"] = "text/csv"
            return resp  
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File (only CSV files accepted)</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    ''' 

 # action={{url_for('tag', dataset='df')}}   

# @app.route('/tag/<dataset>', methods=['POST'])
# def tag(dataset):
# 	#getting our trained model	
# 	model = pickle.load(open("model.pkl", "rb"))   	
# 	input_dataset = request.get_json()
# 	tags = model.predict(input_dataset).to_list()
# 	#Add one row to the dataset
# 	for i in range(len(tags)):
# 		input_dataset.loc1[1: i] = tags[i]
# 	response = {}
# 	response['tags'] = tags
# 	#returning the response object as json
#     return flask.jsonify(response)

if __name__ == '__main__':
     app.run(debug=True)

     




