from flask import Flask, redirect, request, render_template, url_for, send_from_directory, make_response
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
#from jsonrpc import JSONRPCResponseManager, dispatcher
from jsonrpc import dispatcher
import os
import hashlib
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import pickle
import re
from sklearn.neural_network import MLPClassifier
from fasttext import load_model #used to be fastText
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

def separate_words_cap(lst):
	#separate words according to capitalization. Ex: projectNumber --> project Number
    wordsFiltered = []
    for word in lst:
        temp = re.sub( r"([A-Z])", r" \1", word).split()
        if not temp or (len(temp) == len(word)):
            wordsFiltered.append(word)
        else:     
            string = ' '.join(temp)
            wordsFiltered.append(string)
    return wordsFiltered    

def clean_cols(data):
	data = separate_words_cap(data)
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

def select_randomly(dataset, threshold = 200):
    #ensures that dataset isn't skewed towards particular tags by ensuring that each tag has at most a given number 
    #(default: 200) of rows defined by the threshold
    new_dataset = dataset
    tags_to_be_pruned = dataset['Tag'].value_counts()[dataset['Tag'].value_counts() > threshold].keys().tolist()
    for tag in tags_to_be_pruned:
        count = dataset['Tag'][dataset['Tag'] == tag].value_counts()
        drop_count = count - threshold
        drop_count = drop_count.tolist()[0]
        new_dataset = new_dataset.drop(np.random.choice(new_dataset[new_dataset['Tag']==tag].index,size=drop_count,replace=False))
    return new_dataset    

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

def check_mappings(headers, predicted_tags):
    count = 0
    index = 0
    
    #MAPPINGS - SUBJECT TO MODIFICATION
    #Only changed if the confidence probability < 0.8
    MAPPINGS = {
    "#geo" : ['lon', 'lat'], #words that would likely appear for #geo tag
    "#admin" : ['county'], #words that would likely appear for #admin tag
    "#country" :  ['country'], #words that would likely appear for #country tag
    "#date" : ['year', 'date'], #words that would likely appear for #date tag
    "#funding": ['funding', 'funded'], #words that would likely appear for #funding tag
    "#value": ['percentfunded'], #words that would likely appear for #value tag
    "#org":['organization', 'funder ref'], #words that would likely appear for #org tag
    "#status":['status'] #words that would likely appear for #status tag
    }
    for header in headers:
        predicted_tag = predicted_tags[index]
        for key, val in MAPPINGS.items():
            #check if the header contains any of the words in the mappings (substrings are included)
            if (any(header in mystring for mystring in val)):
                if (predicted_tag != key):
                    predicted_tags[index] = key
                    count += 1
        index += 1
    return count, predicted_tags

#post-processing function that 1) checks tags with low confidence against mappings 2) fills in a blank prediction for 
#tags with a confidence level lower than threshold and had no obvious mappings associated with the predicted tag. 
#The function returns 1) the count of corrected tags 2) predicted and predicted tags

def check_mapping(header, predicted_tag):
    MAPPINGS = {
    "#geo" : ['lon', 'lat', 'latitude', 'longitude'], #words that would likely appear for #geo tag
    "#admin" : ['county'], #words that would likely appear for #admin tag
    "#country" :  ['country'], #words that would likely appear for #country tag
    "#date" : ['year', 'date'], #words that would likely appear for #date tag
    "#funding": ['funding', 'funded'], #words that would likely appear for #funding tag
    "#value": ['percentfunded'], #words that would likely appear for #value tag
    "#org":['organization', 'funder ref', 'org'], #words that would likely appear for #org tag
    "#status":['status'], #words that would likely appear for #status tag
    "#sector":['sector'], #words that would likely appear for #sector tag
    "#adm1":['adm1', 'admin1'], #words that would likely appear for #adm1 tag
    "#adm2":['adm2', 'admin2'], #words that would likely appear for #adm2 tag
    "#adm3":['adm3', 'admin3'], #words that would likely appear for #adm3 tag
    "#adm4":['adm4', 'admin4']  #words that would likely appear for #adm4 tag        
    }
    change_tag = False
    header_words = header.split()
    for key, val in MAPPINGS.items():
        for word in header_words:
            #check if the header contains any of the words in the mappings (substrings are not included)
            if (word in val):
                if (predicted_tag != key):
                    predicted_tag = key
                    change_tag = True
    return change_tag, predicted_tag
    

def post_processing(headers, predicted_tags, clf, X_test, mapping_threshold = 0.85, blank_threshold = 0.2):
    if (not isinstance(X_test, np.ndarray)):
        X_test = X_test.values.tolist()
    probs = clf.predict_proba(X_test)
    values = []
    corrected_count = 0
    blank_count = 0
    for i in range(len(X_test)):
        max_arg = probs[i].argsort()[-1]
        top_suggested_tag = clf.classes_[max_arg]
        prob = np.take(probs[i], max_arg)
        predicted_tag = predicted_tags[i]
        if (prob < mapping_threshold):
            header = headers.tolist()[i]
            inc, predicted_tag = check_mapping(header, predicted_tag)
            if (inc):
                corrected_count += 1
            else:
                if (prob < blank_threshold): 
                    predicted_tag = ''
                    blank_count += 1
        values.append(predicted_tag)
    return corrected_count, blank_count, values 

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
        #NEW CODE
        predicted_tags = output_dataset.iloc[:, 0].values
        headers = raw['Header'] 
        predicted_tags = add_hashtags(predicted_tags)
        corrected_count, blank_count, predicted_tags = post_processing(headers, predicted_tags, model, processed_dataset["features_combined"])
        #count, predicted_tags = check_mappings(headers, predicted_tags)
        output_dataset = pd.DataFrame(predicted_tags)
        ###
		#output_dataset = fill_blank_tags(predicted_tags, model, processed_dataset["features_combined"], raw['Header'])
		#output_dataset = pd.DataFrame(add_hashtags(output_dataset))
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

     




