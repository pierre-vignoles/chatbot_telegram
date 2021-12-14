import json
import spacy
import pickle
import os
from typing import Dict, List, Union, Tuple
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from spacy import Language

from my_config import vocab_size, max_len


def function_load_json(path: str) -> Tuple[
    Dict[str, List[Dict[str, Union[str, List[str]]]]], List[str], int, List[int], LabelEncoder]:
    with open(path) as file:
        data = json.load(file, strict=True)

    training_sentences: List[str] = []
    training_labels_str: List[str] = []
    labels: List[str] = []
    responses: List[List[str]] = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels_str.append(intent['tag'])
        responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    num_classes: int = len(labels)

    lbl_encoder: LabelEncoder = LabelEncoder()
    lbl_encoder.fit(training_labels_str)
    training_labels: List[int] = lbl_encoder.transform(training_labels_str)

    return data, training_sentences, num_classes, training_labels, lbl_encoder


def init_nlp() -> Tuple[spacy.Language, set]:
    # load Spacy english
    nlp = spacy.load("en_core_web_sm")
    # load stop words
    stopWords = nlp.Defaults.stop_words
    return nlp, stopWords


def nlp_pipeline(nlp: spacy.Language, stopWords: set, sentences: List[str]) -> Tuple[List[List[int]], Tokenizer]:
    list_sentences: List[List[str]] = []
    token_list: List[str] = []
    oov_token: str = "<OOV>"

    # SPACY
    for text in sentences:
        doc = nlp(text)
        for token in doc:
            if token not in stopWords:
                token_list.append(token.lemma_)
        token_list = " ".join(token_list)
        list_sentences.append(token_list)
        token_list = []

    # KERAS
    # tokenizer => vectorize all the words in the training dataset
    if os.path.isfile('model/tokenizer.pickle'):
        # Loading
        print("Tokenizer already exists. Load it..\n")
        with open('model/tokenizer.pickle', 'rb') as handle:
            tokenizer: Tokenizer = pickle.load(handle)
    else:
        print("Tokenizer does not exist. Create it..\n")
        tokenizer: Tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        tokenizer.fit_on_texts(list_sentences)
        # Saving
        with open('model/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    word_index: Dict[str, int] = tokenizer.word_index
    sequences: List[List[int]] = tokenizer.texts_to_sequences(list_sentences)
    # pad_sequences => allow to set all the training text sequences to the same size
    padded_sequences: List[List[int]] = pad_sequences(sequences, truncating='post', maxlen=max_len)

    return padded_sequences, tokenizer


def function_transform_input_user(input: str, tokenizer, stopWords: set, nlp: Language) -> List[int]:
    token_list: List[str] = []
    # SPACY
    doc = nlp(input)
    for token in doc:
        if token not in stopWords:
            token_list.append(token.lemma_)
    token_list: str = " ".join(token_list)

    # KERAS
    sequence: List[int] = pad_sequences(tokenizer.texts_to_sequences([token_list]), truncating='post', maxlen=max_len)

    return sequence
