import pandas as pd
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from fastText import load_model
from Model import process_dataset_2

#testing MLP Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

fasttext_model = 'wiki.en.bin'
fmodel = load_model(fasttext_model)
# print("Pre-trained model loaded successfully!\n")


col_names = ['Header', 'Tag', 'Attributes','Data','Relative Column Position','Dataset_name', 'Organization','Index']
headers_and_tags= pd.DataFrame(columns = col_names)

# for i in range(5):
rand_dataset = np.random.randint(0, len(datasets_HXL))
process_dataset_2(datasets_HXL[rand_dataset], 'CSV', headers_and_tags, './datasets', count)


#Classification using only headers
df = headers_and_tags
df['Header_embedding'] = df['Header'].map(lambda x: fmodel.get_sentence_vector(str(x)))
df['Organization_embedded'] = df['Organization'].map(lambda x: fmodel.get_sentence_vector(str(x)))
print("Word embeddings extracted!\n")

X_train, X_test, y_train, y_test = train_test_split(df['Header_embedding'], 
                                                    df['Tag'], test_size=0.33, random_state=0)
clf = MLPClassifier(activation='relu', alpha=0.001, epsilon=1e-08, hidden_layer_sizes=1, solver='adam')
clf.fit(X_train.values.tolist(), y_train.values.tolist())
clf.predict(X_test.values.tolist())


#exporting my model
pickle.dump(clf,open("model.pkl","wb"))

# #checking for error
# ans = regr.predict(X_test)
# print mean_squared_error(y_test, ans)
