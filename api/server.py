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
from nltk import ngrams
import json

from nltk.corpus import stopwords
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
#Use server.ipynb for testing in a Jupyter notebook.

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
ALLOWED_EXTENSIONS_CSV = set(['csv'])
ALLOWED_EXTENSIONS_JSON = set(['json'])

app = Flask(__name__)
# CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(app.instance_path)

def lower_cols(lst):
    #convert data to lowercases
    #QUESTION: will I miss any important information? 
    return [word.lower() for word in lst if isinstance(word,str)]


def remove_chars(lst):
    #remove punctuation characters such as ",", "(", ")", """, ":", "/", and "."
    #NOTE: PRESERVES WHITE SPACE.
    #QUESTION: any other characters we should be aware of? Is this a good idea? I'm inspecting each word individually.
    #Any potential pitfalls? 
    cleaned = [re.sub('\s+', ' ', mystring).strip() for mystring in lst if isinstance(mystring,str)]
    cleaned = [re.sub(r'[[^A-Za-z0-9\s]+]', ' ', mystr) for mystr in cleaned if isinstance(mystr,str)]
    cleaned = [mystr.replace('_', ' ') for mystr in cleaned]
    return cleaned

def clean_cols(data):
    data = lower_cols(data)
    data = remove_chars(data)
    return data

def fill_empty_cols(df):
    #Adds 1 in the 2nd row of an empty column.
    empty_cols = []
    for i in df.columns.values:
        if (len(df[i].dropna()) == 0):
            df.at[2,i] = 1
            empty_cols.append(df.columns.get_loc(i))
    return df, empty_cols

def preprocess(pandas_dataset, df_target):
    if (not pandas_dataset.empty):
        organization = 'HDX'   #Replace if datasets contains organization
        headers = list(pandas_dataset.columns.values)
        #Drops rows with all nan values.
        pandas_dataset.dropna(how = 'all', inplace = True)
        pandas_dataset, empty_cols = fill_empty_cols(pandas_dataset)
        #Drops columns with all nan values. Subset parameter is used to exclude column label.
        pandas_dataset.dropna(axis=1, how = 'all', subset=range(1,len(pandas_dataset)), inplace = True)
        headers = clean_cols(headers)
    for i in range(len(headers)):
        #Makes a dictionary of each header and its data and adds these dictionaries to dataframe df_target.
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
    return df_target, df_result, empty_cols

def transform_vectorizers(df_target):
    number_of_data_point_to_vectorize = 7
    cols = ['Header_embedding', 'Organization_embedded', 'features_combined']
    df = pd.DataFrame(columns = cols)
    df_target.dropna(how = 'all', inplace = True)
    df_target, number_of_data_point_to_vectorize = embedded_datapoints(df_target, 7)
    df['data_combined'] = df_target.loc[:, 'embedded_datapoint0': 'embedded_datapoint' 
                                                           + str(number_of_data_point_to_vectorize-1)].values.tolist()
    df['data_combined'] = df['data_combined'].apply(lambda x: [val for item in x for val in item])
    #Uses FastText to generate header and orgaanization embeddings
    df['Header_embedding'] = df_target['Header'].astype(str).apply(fmodel.get_sentence_vector)
    df['Organization_embedded'] = df_target['Organization'].astype(str).apply(fmodel.get_sentence_vector)
    cols = ['Header_embedding', 'Organization_embedded', 'data_combined']
    df['features_combined'] = df[cols].values.tolist()
    df['features_combined'] = df['features_combined'].apply(lambda x: [val for item in x for val in item])
    #2700 is the number of rows needed to prevent ShapeErrors when running the prediction model.
    #The code below appends 0s to prevent these errors but not sure if these 0s affect the predictions.
    diff = 2700 - len(df['features_combined'][0])
    for i in range(len(df)):
        for j in range(diff):
            df['features_combined'][i].append(0)
    df = df.dropna()
    return df


def separate_words(series): 
    #each series is a long string that contains all the data
    lst = []
    cleanlist = [str(x) for x in series if str(x) != 'nan']
    for i in cleanlist:
        lst = re.split(r"\W+", i)
        lst.extend(list(filter(None, lst)))
    return lst

    
def vectorize_n_datapoints(df, number_of_datapoints_to_vectorize = 7):
#     print(df['Data'].head())
#     print(df['Data'].iloc[0])
#     for i in range(len(df['Data'])):
#         df['Data_separated'].iloc[0] = separate_words(df['Data'].iloc[0])
    df['Data_separated'] = df['Data'].apply(separate_words)
    if (number_of_datapoints_to_vectorize > len(df['Data_separated'][0])):
        number_of_datapoints_to_vectorize = len(df['Data_separated'][0])
    for i in range(number_of_datapoints_to_vectorize):
        df['datapoint' + str(i)] = df['Data_separated'].str[i]
    return df, number_of_datapoints_to_vectorize


def embedded_datapoints(df, number_of_data_point_to_vectorize=7):
    df, number_of_data_point_to_vectorize = vectorize_n_datapoints(df)
    print(df.head())
    for i in range(number_of_data_point_to_vectorize):
        #Uses FastText to generate vector word embeddings
        df['embedded_datapoint' + str(i)] = df['datapoint' + str(i)].map(lambda x: fmodel.get_sentence_vector(str(x)))
    return df, number_of_data_point_to_vectorize


def remove_stop_words(data_lst):
    #remove stopwords from the data including 'the', 'and' etc.
    wordsFiltered = []
    for w in data_lst:
        if w not in stopWords:
            wordsFiltered.append(w)
    return wordsFiltered


def word_extract(row):
    ignore = ['nan']
    no_white = [i.lstrip() for i in row if i not in ignore and not (isinstance(i, float) or isinstance(i,int))]
    cleaned_text = [w.lower() for w in no_white if w not in ignore]
    return cleaned_text

def allowed_file_csv(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_CSV

def allowed_file_json(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_JSON

def generate_n_grams(data_lst, n):
    # cleaned = remove_chars(list(data_lst))
    # cleaned = clean_cols(cleaned)
    cleaned = remove_stop_words(data_lst)
    #make sure that n_grams 'refresh' when a new dataset is encountered!!!!   
    return list(ngrams(cleaned, n))


def tag_predicted(clf, X_test, series, threshold):
    #True if tag should be left blank
    if (not isinstance(X_test, np.ndarray)):
        X_test = X_test.values.tolist()
    probs = clf.predict_proba(X_test)
    values = []
    for i in range(len(X_test)):
        max_arg = probs[i].argsort()[-1]
        top_suggested_tag = clf.classes_[max_arg]
        prob = np.take(probs[i], max_arg)
        if (prob > threshold):
            values.append(False)
        else:
            values.append(True)
    return values

#helper function to fill in the blanks for tags that have a confidence level less than the threshold
def fill_blank_tags(predicted_tags, clf, X_test, series, threshold = 0.3):
    boolean_array = tag_predicted(clf, X_test, series, threshold)
    for i in range(len(predicted_tags)):
        if (boolean_array[i] == True):
            predicted_tags[i] = ''
    return predicted_tags

def add_hashtags(predicted_tags):
    result = []
    if (isinstance(predicted_tags, np.ndarray)):
        for word in predicted_tags:
            if word == '':
                result.append('')
            else:
                result.append("#"+word)
    return result

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            # flash('No selected file')
            return redirect(request.url)
        # file.save(os.getcwd())
        if file and allowed_file_csv(file.filename):
            filename = secure_filename(file.filename)
            input_dataset = pd.read_csv(file)
                

        if file and allowed_file_json(file.filename):
            # filename = secure_filename(file.filename)
            input_dataset = pd.read_json(file)
            input_dataset = input_dataset.rename(columns=input_dataset.iloc[0]).drop(input_dataset.index[0])
                # process the untagged dataset
        input_headers = input_dataset.columns.values
        raw, processed_dataset, empty_cols = preprocess(input_dataset, 
                               pd.DataFrame(columns=['Header','Data','Relative Column Position','Organization','Index']))
        model = pickle.load(open("model.pkl", "rb")) #Model needs be named model.pkl, preferably using version 0.20.3
        output_dataset = pd.DataFrame(data = model.predict(list(processed_dataset['features_combined'])))
        output_dataset.loc[empty_cols,0] = ''
        output_dataset = fill_blank_tags(output_dataset.iloc[:, 0].values, model, processed_dataset["features_combined"], raw['Header'])
        output_dataset = pd.DataFrame(add_hashtags(output_dataset))
        input_dataset.loc[-1] = output_dataset.iloc[:, 0].values
        input_dataset.index = input_dataset.index + 1
        input_dataset = input_dataset.sort_index()

        resp = make_response(input_dataset.to_csv())
        resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
        
          


    return '''
    <!doctype html>
    <head>
    <title>Upload new File</title>
 
    </head>
    <body style="background-color:#91C9E8; font-family: "Times New Roman", Times, serif";>
    <div style="position: relative; left: 300px; top: 200px;">
    <h3>Upload the dataset that you want to add HXL tags, </h3>
    <h3>and a file with tagged dataset will be downloaded. </h3>
    <h3> (only CSV and JSON files accepted)</h3>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <div>
    <body>
    ''' 


if __name__ == '__main__':
     app.run(debug=True)

     




