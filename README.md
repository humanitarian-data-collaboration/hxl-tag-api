# hxl-tagger
API using Flask to predict hxl tags in a csv.

To get started:

0. Clone this repository, and create a virtual environment with Python3

1. Install fastText for header embeddings
   - `git clone https://github.com/facebookresearch/fastText.git`
   - `cd fastText`
   - `pip install .`
 
2. 'wiki.en.bin' needs to be in the same directory as server.py
    - Download from https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
    - Extract from zip file
    - Move to hxl-tag-api/api folder
  
3. Install dependencies with `pip install -r requirements.txt`, preferably in the virtual environment.

4. Use `python server.py` to run the API.

***Known Issues***:

1. The API can't handle files with missing values. Make sure to fill in missing values in the dataset.
2. For now, the API can only handle .csv and .json files.
