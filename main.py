from function_utils import *
from model import *
from my_config import *

from telebot import TeleBot
from telebot import types


def function_message_init_bot() -> str:
    return """Hi, I'm Pierre and this is the bot I created to introduce myself. 
You can ask for information about me.
You can also click on one button below if you are interested in these topics.
Write /help if you want to see these informations again."""


def function_regroup_all() -> Tuple[
    Sequential, Tokenizer, LabelEncoder, set, Language, Dict[str, List[Dict[str, Union[str, List[str]]]]]]:
    data, training_sentences, num_classes, training_labels, lbl_encoder = function_load_json(path_intents)
    nlp, stopWords = init_nlp()
    padded_sequences, tokenizer = nlp_pipeline(nlp, stopWords, training_sentences)
    model = function_model_training(nb_epochs, padded_sequences, training_labels, num_classes)

    return model, tokenizer, lbl_encoder, stopWords, nlp, data


if __name__ == '__main__':
    bot = TeleBot(TOKEN, parse_mode=None)

    model, tokenizer, lbl_encoder, stopWords, nlp, data = function_regroup_all()


    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        chatid = message.chat.id

        markup = types.ReplyKeyboardMarkup()
        itembtn1 = types.KeyboardButton('My resume')
        itembtn2 = types.KeyboardButton('My professional experience')
        itembtn3 = types.KeyboardButton('My portfolio')
        itembtn4 = types.KeyboardButton('Contact me')
        markup.row(itembtn1, itembtn2)
        markup.row(itembtn3, itembtn4)

        bot.send_message(chatid, function_message_init_bot(), reply_markup=markup)


    # Scan only text message
    @bot.message_handler(func=lambda message: True, content_types=['text'])
    def echo_all(message):
        chatid = message.chat.id
        markup = types.ReplyKeyboardRemove(selective=False)

        result, answer_valid = function_return_predict_model(message.text, model, tokenizer, stopWords, nlp)
        answer_text, answer_file_link = function_return_type_answer_model(answer_valid, result, lbl_encoder, data)
        if answer_file_link is None:
            bot.send_message(chatid, answer_text, reply_markup=markup)
        if type(answer_file_link) == str:
            bot.send_message(chatid, answer_text, reply_markup=markup)
            doc = open(answer_file_link, 'rb')
            bot.send_document(chatid, doc)
        elif type(answer_file_link) == list:
            bot.send_message(chatid, answer_text[0], reply_markup=markup)
            for idx, photo in enumerate(answer_file_link):
                bot.send_message(chatid, answer_text[idx + 1])
                if "pic" in photo:
                    doc = open(photo, 'rb')
                    bot.send_photo(chatid, doc)
                elif "giphy" in photo:
                    bot.send_animation(chatid, photo)


    # Verification of the content type of the message received : if type other than text => return a specific message
    @bot.message_handler(func=lambda message: True, content_types=['photo', 'audio', 'video', 'document', 'sticker',
                                                                   'video_note', 'voice', 'location', 'contact'])
    def echo_other(message):
        chatid = message.chat.id
        markup = types.ReplyKeyboardRemove(selective=False)
        only_test_message: str = "This chatbot only accepts text messages."
        bot.send_message(chatid, only_test_message, reply_markup=markup)


    bot.infinity_polling(interval=0, timeout=10)
