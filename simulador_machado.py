import os
import random
import telegram
from telegram.ext import Updater, MessageHandler, Filters

import openai

# Configuração das chaves de API
TELEGRAM_TOKEN = ''
OPENAI_API_KEY = ''

# Configuração do OpenAI
openai.api_key = OPENAI_API_KEY

# Carregar textos do Machado de Assis
textos_machado_dir = 'D:/Users/ter95063/Documents/Ferramentas/notebooks/classification/textos_do_machado/'
textos_machado = []

for filename in os.listdir(textos_machado_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(textos_machado_dir, filename), 'r', encoding='utf-8') as file:
            textos_machado.append(file.read())

# Função para gerar resposta do chatbot
def generate_response(prompt):
    response = openai.Completion.create(
        engine="davinci",  # Pode ajustar o mecanismo conforme necessário
        prompt=prompt,
        max_tokens=100  # Ajuste conforme necessário
    )
    return response.choices[0].text.strip()

# Função para lidar com as mensagens do Telegram
def handle_message(update, context):
    user_message = update.message.text
    machado_prompt = f"Eu sou o Machado de Assis, e vou responder sua pergunta:\n\n{user_message}\n\nResposta:"

    bot_response = generate_response(machado_prompt)
    update.message.reply_text(bot_response)

def main():
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
