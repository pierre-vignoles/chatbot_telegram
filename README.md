# Telegram chatbot
Telegram chatbot created with deep learning model (LSTM) and telebot library.

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)  ![badge-telegram-program](https://github.com/pierre-vignoles/chatbot_telegram/blob/master/img/telegram-program.svg)

## Description
This program will allow you to create very easily a custom chatbot capable of sending texts, images or videos.  
The deep learning model used is a Long Short Term Memory (LSTM) but you can change his structure if you want.

## Getting Started
### Install the libraries
Execute the following command : `pip install -r requirements.txt`

### Create a bot in telegram
To do that, you have to open your telegram application. Then speak to this user : @BotFather. It is a bot created by telegram itself that allows you to manage the creation and the editing of your bots.
Just follow the instructions and get the API token of your bot.  
You can follow this tutorial : [tutorial](https://core.telegram.org/bots#6-botfather)

### Change the settings
Once you got the api token of your bot, in the `my_config.py` file change the value of `TOKEN`.

## Last step
Before you can run this program, you will have to complete the most important file : `ìntents.json`
In this file you will have to write all the sentences you would like the chatbot learn. 
* tag : title of the question/answer. It does not matter for the model. This is just for you.
* type : You have 4 possibilities
    * text : the chatbot will answer only with a text message
    * file : the chatbot will send a text message followed by a document
    * photos : the chatbot will send a text message followed by a photo.
    * multiple_photos : the chatbot will send text messages with pictures attached.
* patterns : write all possible turns of phrase for a question
* responses : write the chatbot's answers. You can write more than one. The chatbot will choose randomly one of them.
* link : path to your documents and photos

## Launch
Execute the following command in a terminal : `python main.py` 

## Heroku
You can add this program on HEROKU to let it work without interruption for free. The file Procfile is there for that.

### You can now speak to your bot in your telegram application !
