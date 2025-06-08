from flask import Flask, render_template, request
from ai import ai

app = Flask(__name__)

messages = []


@app.route('/', methods = ['GET', 'POST'])
def index():
    user_input = ''
    if request.method == 'POST':
        user_input = request.form.get("message")
    if user_input:
        bot_reply = ai(user_input)
        messages.append(f'Ты:{user_input}')
        messages.append(f'Бот:{bot_reply}')
        print("Пользователь ввёл:", user_input) 
    return render_template('index.html', messages=messages)

if __name__ == '__main__':
    app.run(debug=True)