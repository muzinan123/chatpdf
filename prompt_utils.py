prompt_template = """
You are a question-answering bot.
Your task is to answer user questions based on the given information below.
Make sure your response is entirely based on the information provided below. Do not make up answers.
If the information provided below is not sufficient to answer the user's question, please respond directly with "I cannot answer your question".

Known information:
__INFO__

User asks:
__QUERY__

Please answer the user's question in Chinese.
"""


def build_prompt(template=prompt_template, **kwargs):
    """Assign values to the Prompt template"""
    prompt = template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        prompt = prompt.replace(f"__{k.upper()}__", val)
    return prompt
