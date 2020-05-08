''' Below code accepts the transcribed data and does the following preprocessing :
    -removes stopwords
    -remove special characters
    -converts to lower case
    -lemmatize the words
    -converts numeric to words
After preprocessing it performs count vectorization ( used for own learner implementation) and TF-IDF vectorization (used
with sklearn learner implementation)
Also, cosine similarity matching assessment is done on the entire, 93% accuracy is obtained in matching transcripts's
labels : Spam v/s Non-Spam

    '''

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
import nltk
nltk.download('stopwords')
nltk.download('punkt')
spacy.load('en_core_web_sm')
lemmatizer = spacy.lang.en.English()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from num2words import num2words
import os
import numpy as np
import pandas as pd


########################################### TEXT PREPROCESSING PART BEGINS  ###########################################
#preprocessing functions
def to_lower(data):
    return np.char.lower(data)
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text
def remove_symbols(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data
def rem_apostrophy(data):
    return np.char.replace(data, "'", "")

def numToword(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text
def preprocess(data):
    data = to_lower(data)
    print("converted to lower case")
    data = remove_symbols(data) #remove comma seperately
    print("removed punctuation")
    data = rem_apostrophy(data)
    print("removed apostrophe")
    data = remove_stop_words(data)
    print("removed stop words")
    data = numToword(data)
    print("converted numbers")
    data = remove_symbols(data)
    print("removed punctuation")
    data = numToword(data)
    print("converted numbers")
    data = remove_symbols(data)
    print("removed punctuation")
    data = remove_stop_words(data)
    print("removed stop words")
    return data



#Pass the source transcripts file
text="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\transcripts_master.csv"
df= pd.read_csv(text)
N = len (df)
processed_text = []
for i in df['value']:
    text = i
    print(text)
    processed_text.append(preprocess(text)) # preprocess each row

# print(processed_text)
# len(processed_text)
##################################### TEXT PREPROCESSING PART ENDS  #########################################



##################################### VECTORIZATION PART BEGINS  #####################################

#lemmatization of tokens
def my_tokenizer(doc):
    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens])

# create a dataframe from a word matrix
def wm2df(wm, feat_names):
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return (df)


# Count Vectorization
vectorizer = CountVectorizer(tokenizer=my_tokenizer,ngram_range=(2,2))
wm = vectorizer.fit_transform(processed_text)
tokens=vectorizer.get_feature_names()
cntvec_df=wm2df(wm, tokens) #final vectorized dataframe




#TF-IDF vectorization
vectorizer = TfidfVectorizer(ngram_range=(2,2),norm='l2') # You can still specify n-grams here.
wm = vectorizer.fit_transform(processed_text)

##################################### COSINE SIMILARITY PART BEGINS  #####################################
from sklearn.metrics.pairwise import cosine_similarity
tokens = vectorizer.get_feature_names()
tdidf_df=wm2df(wm, tokens) #final vectorized dataframe


#Cosine Similarity Calculating + Final Cross Tab evalutation to find proportion of correct label matched
mat=np.asmatrix((cosine_similarity(tdidf_df)))
np.fill_diagonal(mat, 0)
# mx=mat[1].argmax(axis=1)
mx=pd.DataFrame(mat)
maxValIndex = pd.DataFrame(mx.idxmax(axis=1),columns=["match"])
maxValIndex.index.name = 'source'
maxValIndex.reset_index(inplace=True)
maxValIndex['Source-Label']=0
maxValIndex['Match-Label']=0
df.iloc[maxValIndex.iloc[0,1],0]
for i in range(0,len(maxValIndex)):
    maxValIndex.iloc[i,2]=df.iloc[maxValIndex.iloc[i,0],0]
    maxValIndex.iloc[i, 3] = df.iloc[maxValIndex.iloc[i, 1], 0]

cross_freq=pd.crosstab(maxValIndex['Source-Label'],maxValIndex['Match-Label'],margins=True)
(cross_freq.iloc[0,0]+cross_freq.iloc[1,1])/(cross_freq.iloc[0,0]+cross_freq.iloc[0,1]+cross_freq.iloc[1,0]+cross_freq.iloc[1,1])
maxValIndex.to_csv("C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\cosine_similarity.csv")
cross_freq.to_csv("C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\cosine_similarity_summary.csv")
##################################### COSINE SIMILARITY PART ENDS  #####################################
#Writing out the tfidf data for further classifiers
df.index = tdidf_df.index
tdidf_df['Spam-Label']=df['Spam-Label']
tdidf_df.index.name = 'Document'
tdidf_df.reset_index(inplace=True)
cols = list(tdidf_df)
cols.insert(1, cols.pop(cols.index('Spam-Label')))
tdidf_df = tdidf_df.loc[:, cols]

output="C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\tfidf_master.pkl"
tdidf_df.to_pickle(output)

##################################### VECTORIZATION PART ENDS  #####################################