from flask import   Flask, render_template,request
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
import re
from tensorflow.keras.models import load_model
# from keras.backend import set_session
# from corona-tweet-sentiment-annalysis import Lemmatizer
# import tensor

class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, sentence):
        sentence=re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)',' ',sentence)
        sentence=re.sub('[^0-9a-z]',' ',sentence)
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word)>1]

# tf_config = some_custom_config
# sess = tf.Session()
# global graph
# graph = tf.get_default_graph()
# set_session(sess)

# lstm_model=pickle.load(open('lstm_model.pickle','rb'))
lstm_lbl_encoder=pickle.load(open('encoder.pickle','rb'))
lstm_tokenizer=pickle.load(open('lstm_tokenizer.pickle','rb'))
lstm_model=load_model('lstm_model.h5')

app= Flask(__name__)
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/text_nlp',methods=['GET', 'POST'])

def sentiment_predict():
    predictions=""
    if request.method =='POST':
        tweet=request.form.get('message')
        print(tweet)
        token=lstm_tokenizer.texts_to_sequences([tweet])
        print(token)
        pad_seq=pad_sequences(token,maxlen=60,padding='post',truncating='post')
        print(pad_seq)
        preds = lstm_model.predict_classes(pad_seq)
        predictions="The above typed tweet is {0}".format(lstm_lbl_encoder.inverse_transform(preds)[0])
        # predictions=lstm_model.predict_classes(pad_seq) # You need to use the following line
        # with graph.as_default():

        #     set_session(sess)
        #     preds = lstm_model.predict_classes(pad_seq)
        #     print(preds)
        # predictions='hello'

    return render_template('text_preprocessing.html',lstm_prediction=predictions)

if __name__=='__main__':
    app.run(debug=True)