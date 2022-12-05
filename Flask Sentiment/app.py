from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

import pickle, re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title' : LazyString(lambda: 'API Documentation for Sentiment Prediction'),
        'version' : LazyString(lambda: '1.0.0'),
        'description' : LazyString(lambda: 'Dokumentasi API untuk Prediksi Sentimen')
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'docs',
            'route': '/docs.json',
        }
    ],
    'static_url_path': "/flasgger_static",
    'swagger_ui': True,
    'specs_route': "/docs/"
}
swagger = Swagger(app, template=swagger_template,             
                  config=swagger_config)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features,split=' ',lower=True)

sentiment = ['negative','neutral','positive']

#Funct Cleansing Text, lowercase & filter only use alphanumeric
def cleansing(sent):

    string = sent.lower()

    string = re.sub(r'[^a-zA-Z0-9]',' ',string)
    return string

#Load Feature Extraction for LSTM
file = open("lstm model/x_pad_sequences.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file.close()
#Load Model for LSTM
model_file_from_lstm = load_model('lstm model/model.h5')

#Load Feature Extraction for Neural Network
count_vect = pickle.load(open("Neural Network Model/feature.p","rb"))
#Load Model for Neural Network
model = pickle.load(open("Neural Network Model/model.p","rb"))


#Func API Neural Network
@swag_from("docs/nn-file.yml",methods=['POST'])
@app.route('/nnfile',methods=['POST'])
def nnfile():
    # Upladed file
    file = request.files['file']

    # Import file csv ke Pandas
    df = pd.read_csv(file,header=None,encoding='latin-1')
    df= df.rename(columns={0: 'text'})
    df['text_clean'] = df.apply(lambda row : cleansing(row['text']), axis = 1)
    
    result = []

    for index, row in df.iterrows():
        # Feature Extraction
        text = count_vect.transform([(row['text_clean'])])

        # Kita prediksi sentimennya
        result.append(model.predict(text)[0])

    # Get result from file in "List" format
    original = df.text.to_list()

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using Neural Network",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }

    response_data = jsonify(json_response)
    return response_data


#Func API Neural Network
@swag_from("docs/nn.yml",methods=['POST'])
@app.route('/nn',methods=['POST'])
def nn():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]

    # Feature Extraction
    text = count_vect.transform(text)

    # Kita prediksi sentimennya
    result = model.predict(text)[0]

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using Neural Network",
        'data' : {
            'text' : original_text,
            'sentiment' : result
        },
    }

    response_data = jsonify(json_response)
    return response_data

#Func API LSTM
@swag_from("docs/lstm.yml",methods=['POST'])
@app.route('/lstm',methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'text' : original_text,
            'sentiment' : get_sentiment
        },
    }

    response_data = jsonify(json_response)
    return response_data

#Func API LSTM File
@swag_from("docs/lstm-file.yml",methods=['POST'])
@app.route('/lstmfile',methods=['POST'])
def lstmfile():
    # Upladed file
    file = request.files['file']

    # Import file csv ke Pandas
    df = pd.read_csv(file,header=None,encoding='latin-1')
    df = df.rename(columns={0: 'text'})
    df['text_clean'] = df.apply(lambda row : cleansing(row['text']), axis = 1)
    
    result = []

    for index, row in df.iterrows():
        # Feature Extraction
        text = tokenizer.texts_to_sequences([(row['text_clean'])])
        text = pad_sequences(text,maxlen=feature_file_from_lstm.shape[1])
        prediction=model_file_from_lstm.predict(text)
        # Kita prediksi sentimennya
        get_sentiment=sentiment[np.argmax(prediction[0])]
        result.append(get_sentiment)

    # Get result from file in "List" format
    original = df['text_clean'].to_list()
    
    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__' :
    app.run(debug=True)