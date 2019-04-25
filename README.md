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

1. For now, the API can only handle .csv and .json files. (If using a JSON file, it must follow the format [here](https://proxy.hxlstandard.org/data.json?url=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2F1R9zfMTk7SQB8VoEp4XK0xAWtlsQcHgEvYiswZsj9YA4%2Fedit%23gid%3D0))
