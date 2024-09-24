import openai
import os
# Load environment variables
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # Read local .env file, which defines OPENAI_API_KEY

openai.api_key = os.getenv('OPENAI_API_KEY')


def get_completion(prompt, context, model="gpt-3.5-turbo"):
    """Wrapper for the OpenAI API"""
    messages = context + [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # Randomness of model output, 0 means minimum randomness
    )
    return response.choices[0].message["content"]


def get_embedding(text, model="text-embedding-ada-002"):
    """Wrapper for OpenAI's Embedding model API"""
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
