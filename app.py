import pickle
import re
import string
import nltk as nltk
import pandas as pd
import json
from flask import Flask, url_for, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import jsonify
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from pprint import *

with open('model.pkl', 'rb') as file:
    vect, clf = pickle.load(file)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)


@app.route('/')
def root():
    return "Hello World"


def text_process(mess):
    no_punc = [char for char in mess if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    review = re.sub('[^a-zA-Z]', ' ', no_punc)
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    return review


def remove_stopwords_spam(text):
    stop_words = stopwords.words("english") + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text


@app.route('/spam', methods=['POST'])
def spam():
    if request.method == "POST":
        json_data = json.loads(request.data)
        # print(json_data["comments"])
        data = pd.DataFrame(json_data)
        data['clean_msg'] = data["comments"].apply(text_process)
        data["tokenize_text"] = data.apply(lambda row: nltk.word_tokenize(row["clean_msg"]), axis=1)
        data["No_stopword_Text"] = data["tokenize_text"].apply(remove_stopwords_spam)
        print(data)
        data["v2"] = data['No_stopword_Text'].apply(lambda x: ' '.join(x))
        # print(data)
        x_test = data['v2']
        test_transform = vect.transform(x_test)
        predicted = clf.predict(test_transform)
        data["pred"] = predicted
        result = data["pred"].map({0: 'ham', 1: 'spam'}).to_list()
        return json.dumps(result)


def generate_stop():
    from nltk.corpus import stopwords
    import string
    stop = stopwords.words('english')
    punctuations = list(string.punctuation)
    stop = stop + punctuations
    return stop


def processed_data(text, stop_remove, more_remove, stem, lemme):
    from nltk.tokenize import word_tokenize
    text = text.lower()  # first we convert the text into lower (case senstivite )
    text = word_tokenize(text)  # for splitting into words , as removing the stop words will be done in in words.

    if (stop_remove == True):
        stop = generate_stop()  # by using generate function , we are combining both the stop words and puncations
        text = [w for w in text if not w in stop]  # we remove them by using this .

    if (
            more_remove == True):  # like some of the text has more special charcater like '',`` , these are not the part of stop .
        text = (" ".join(text))
        text = text.replace('``', " ")
        text = text.replace("''", " ")
        text = word_tokenize(text)

    if (stem == True):  # if u want the stemming of your data
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        text = [ps.stem(w) for w in text]

    if (lemme == True):  # To lemmization of the data.
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(w) for w in text]

    text = (" ".join(text))  # joining of the text .
    return text


def sentence_score(text):
    sid = SentimentIntensityAnalyzer()  # making the object of the sentimentanalyser
    sid_dict = sid.polarity_scores(text)  # getting the scores ductionary
    scores = sid_dict['compound']  # taking the compound scores

    if (scores >= 0.05):  # based on the compound score finding the sentiments of the review
        return "positive"
    elif (scores <= -0.05):
        return "negative"
    else:
        return "netural"


@app.route('/sentiment', methods=['POST'])
def semtiment():
    if request.method == "POST":
        json_data = json.loads(request.data)
        # print(json_data["comments"])
        df = pd.DataFrame(json_data)
        df.dropna(subset=['comments'], inplace=True)
        df['processed'] = df['comments'].apply(processed_data, args=(False, True, False, True))
        df['sentiments'] = df['processed'].apply(sentence_score)
        return jsonify(df['sentiments'].to_list())


def remove_punc(text):
    text = text.lower()  # first we convert the text into lower (case senstivite )
    text = word_tokenize(text)
    punctuations = list(string.punctuation)
    text = [w for w in text if not w in punctuations]
    text = (" ".join(text))  # joining of the text .
    return text


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


@app.route('/topics', methods=['POST'])
def topics():
    if request.method == "POST":
        json_data = json.loads(request.data)
        df = pd.DataFrame(json_data)
        df.dropna(subset=['comments'], inplace=True)
        df['processed'] = df['comments'].apply(remove_punc)
        review = df.processed.values.tolist()
        data_words = list(sent_to_words(review))
        data_words = remove_stopwords(data_words)

        # Create Dictionary
        id2word = corpora.Dictionary(data_words)

        # Create Corpus
        texts = data_words

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # remove stop words
        data_words = remove_stopwords(data_words)

        # # number of topics
        num_topics = 20

        # Build LDA model
        lda_model20 = gensim.models.LdaMulticore(corpus=corpus,
                                                 id2word=id2word,
                                                 num_topics=num_topics)

        topics = pformat(lda_model20.show_topics())

        print(topics)
        return topics


if __name__ == "__main__":
    app.run(debug=True)