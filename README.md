# hxl-tagger
API using Flask to predict hxl tags in a csv.

To get started:

1. Pip install fastText by cloning (https://github.com/facebookresearch/fastText.git), cd into the directory and run 'pip install .'<br>
2. 'wiki.en.bin' needs to be in the same directory as server.py (can be downloaded from https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip)
3. Run pip install requirements.txt, preferably in a virtualenv.
4. Use 'python server.py' to run the API.

***Known Issues***:

1. The API can't handle files with missing values. Make sure to fill in missing values in the dataset.
2. For now, the API can only handle .csv files. We will add JSON functionality shortly.
3. The model will not generate predicted tags for the last 2 columns of the dataset for now. This will be fixed shortly.
