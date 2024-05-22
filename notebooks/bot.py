import pickle

import telebot
from main import train, predict, preprocess_text
import sklearn

my_token = 'token'
bot = telebot.TeleBot(my_token)


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, f'Hello, {message.from_user.first_name}! Send your review')


@bot.message_handler(func=lambda message: True)
def review(message):
    try:
        model = 'model.pkl'
        with open(model, "rb") as f:
            my_model = pickle.load(f)
        pred = my_model.predict([preprocess_text(message.text)])
        # print(pred)
        if pred[0] == 1:
            pred_text = 'Your review is positive:)'
        elif pred[0] == 0:
            pred_text = 'Your review is negative:('
        bot.send_message(message.chat.id, pred_text)
        # print(format(sklearn.__version__))
    except Exception as e:
        print(f'{e}')


bot.polling(none_stop=True)
