import numpy as np
import os
from spacy import Language
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, LSTM, Flatten, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from typing import List, Dict, Tuple, Union

from function_utils import function_transform_input_user
from my_config import vocab_size, embedding_dim, max_len


def function_model_training(nb_epochs: int, padded_sequences, training_labels, num_classes: int) -> Sequential:
    if os.path.isfile('model/model.h5'):
        print("Model already exists. Load it..\n")
        model = load_model('model/model.h5')
    else:
        print("Model does not exist. Create it..\n")
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        model.add(LSTM(150, return_sequences=True))
        model.add(LSTM(150))
        model.add(Flatten())
        # model.add(GlobalAveragePooling1D())
        # model.add(Dense(units=128, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        epochs = nb_epochs
        history = model.fit(x=padded_sequences, y=np.array(training_labels), epochs=epochs)
        model.save('model/model.h5')
        # model_loss = pd.DataFrame(model.history.history)
        # model_loss.plot()

    return model


def function_return_predict_model(input_sentence: str, model: Sequential, tokenizer: Tokenizer, stopWords: set,
                                  nlp: Language) -> Tuple[List[float], bool]:
    result = model.predict(function_transform_input_user(input_sentence, tokenizer, stopWords, nlp))
    for proba in result[0]:
        if proba >= 1e-15:
            answer_valid = True
        else:
            answer_valid = False
    return result, answer_valid


def function_return_type_answer_model(answer_valid: bool, result: List[float], lbl_encoder: LabelEncoder,
                                      data: Dict[str, List[Dict[str, Union[str, List[str]]]]]) -> Tuple[str, Union[None, str, List[str]]]:
    if answer_valid == True:
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        for i in data['intents']:
            if i['tag'] == tag:
                if i['type'] == 'text':
                    answer_text = np.random.choice(i['responses'])
                    answer_file_link = None
                elif i['type'] == 'file':
                    answer_text = i['responses']
                    answer_file_link = i['link']
                elif i['type'] == 'multiple_photos':
                    answer_text = i['responses']
                    answer_file_link = i['link']

    else:
        answer_text = np.random.choice(
            ["I'm sorry I do not undertand", "What did you say ?", "Can you rephrase your sentence differently?"])
        answer_file_link = None

    return answer_text, answer_file_link
