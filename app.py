from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# load the tokenizer and model


model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

chat_template = """
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}

"""

#create the jinja2 template
tokenizer.chat_template = chat_template

def vanilla_chatbot(message, history = []):
    # Prepare the input
    chat_history = []

    for human, assistant in history:
        chat_history.append({"role": "user", "content": human})
        chat_history.append({"role": "assistant", "content": assistant})

    chat_history.append({"role": "user", "content": message})

    #tokenize the input

    inputs = tokenizer.apply_chat_template(chat_history, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=1000,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    #Decode the output

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response



demo = gr.ChatInterface(
    vanilla_chatbot,
    title="Murielle Chatbot",
    description="Enter your messange and get an answer from the bot",

)

demo.launch(debug= True, share=True)