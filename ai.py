from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_name = "sberbank-ai/rugpt3large_based_on_gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chat_history = (
    "Ты: Что такое глобальное потепление?\n"
    "Бот: Это повышение температуры Земли из-за парниковых газов.\n"
    "Ты: Как помочь планете?\n"
    "Бот: Сократить использование ископаемого топлива и перерабатывать отходы.\n"
    "Ты: Привет!\n"
    "Бот: Привет! Задавай вопросы об экологии.\n"
    "Ты: Как перерабатывать мусор?\n"
    "Бот: Нужно отсортировывать мусор и сдать в специальное место для переработки\n"
    "Ты: Как не загрязнять планету?\n"
    "Бот: Нужно не мусорить и использовать тип транспорта который не выделяет газы\n"
)

print("Привет! Спрашивай что угодно.")

    
def ai(text):
    global chat_history
    prompt = (
    "Ты — экспертный ЭКО-бот. Отвечаешь СТРОГО на вопросы по экологии, переработке отходов, защите окружающей среды. Отвечай развернуто, конкретно и по делу, давая рекомендации по сортировке, переработке и уменьшению отходов.\n"
    "ЕСЛИ вопрос не по теме — скажи 'Я могу отвечать только на вопросы об экологии.'\n"
    "НИКОГДА не говори о медицине, политике, законах, психологии, здоровье.\n"
    "Говори строго, ясно и по делу.\n"
    + chat_history
    + f"Ты: {text}\nБот:"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=80,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        repetition_penalty=1.2,
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    bot_response = response[len(prompt):].strip()
    bot_response = bot_response.split(".")[0] + "."

# Отрезаем все, что идет после "Ты:" (если есть)
    if 'Ты' in bot_response:
        bot_response = bot_response.split("Ты:")[0].strip()

# Отрезаем все, что идет после "Бот:" (если есть)
    if "Бот:" in bot_response[5:]:
        bot_response = bot_response.split("Бот:")[0].strip()

    print("Бот:", bot_response)

    chat_history += f"Ты: {text}\nБот: {bot_response}\n"
    return bot_response
