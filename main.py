
"""
This script helps answer creator customer support questions, using a combination of SentenceTransformer embeddings,
FAISS for nearest-neighbor search, and OpenAI's GPT model for natural language understanding and generation.

The knowledge base of support questions and answers is loaded from a JSON file (created by get_knowledgebase_json.py). Each question in the knowledge base is transformed into an embedding vector using the SentenceTransformer model so that we can give GPT only the relevant section for a question and manage the number of tokens used. These embeddings are either loaded from a pre-existing file or generated and then saved for future use. A FAISS index is also created or loaded from file to give us an efficient way to search the knowledgebase.

During operation, the script enters a loop, prompting the user to enter a question. The question is transformed into an embedding and the nearest question in the knowledge base is found using the FAISS index. The corresponding answer is then used as part of a prompt to OpenAI's GPT model. The response from the GPT model is considered the answer to the user's question.

The user's question and the generated answer are added to the conversation history, but if it gets over the token limit defined in the config file, the program will forget earlier messages to stay within the limit.

To quit the script, the user may type 'exit' during the question prompt.

Note: This script requires a config.yaml file containing the following keys:
- 'knowledge_base_file': The file path to the knowledge base JSON file.
- 'sentence_transformer_model': The name of the SentenceTransformer model to use.
- 'question_embeddings_file': The file path where question embeddings are/should be stored.
- 'index_file': The file path where the FAISS index is/should be stored.
- 'openai_model': The name of the OpenAI GPT model to use.
- 'token_limit': The maximum length of the conversation history, in tokens.
- 'openai_api_key': api key
"""


import json
from sentence_transformers import SentenceTransformer
import faiss
import openai
import numpy as np
import os
from token_counter import tiktoken_len
import yaml
import re
import html
from dotenv import load_dotenv

load_dotenv()

# Configurable prompt template for setting the behavior of the GPT model
SYSTEM_CONTEXT = """
You are a helpful customer support assistant for Cohley. 
Please answer the question based on the following set of questions and answers. 
If you do not have the relevant information, say so, and do not make up an answer.
---
"""

# Load the configuration file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# gets our openai api key from the environment variables
api_key_env_name = config.get("openai_api_key")
api_key = os.environ.get(api_key_env_name)

if not api_key:
    raise ValueError(f"Please set the environment variable '{api_key_env_name}' with your OpenAI API key.")

# gives the api key to the model
openai.api_key = api_key

# loads Q&A info from knowledgebase json
def load_knowledge_base(file):
    try:
        with open(file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
        return None

# loads or creates embeddings so only relevant Q&A info is passed to GPT
def load_or_create_embeddings(file, questions, model):
    if os.path.isfile(file):
        return np.load(file)
    else:
        embeddings = model.encode(questions)
        np.save(file, embeddings)
        return embeddings

# creates an index file so embeddings are searchable
def load_or_create_index(file, embeddings):
    d = embeddings.shape[1]
    if os.path.isfile(file):
        return faiss.read_index(file)
    else:
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss.write_index(index, file)
        return index

# Helper function used to clean html content when its retrieved from the knowledgebase so we don't waste tokens 
def clean_content(raw_text):
    """Cleans content by removing HTML tags and decoding HTML entities."""
    # Remove HTML tags
    cleanr = re.compile('<.*?>')
    text_without_html = re.sub(cleanr, '', raw_text)
    # Decode HTML entities
    return html.unescape(text_without_html)

# GPT gets the relevant info chunks from the search index and uses this to answer questions
def ask_gpt(user_question, past_messages, model, index, knowledge_base):
    # Convert the user question into an embedding
    question_embedding = model.encode(user_question)

    # Use Faiss to find the closest chunk embeddings. Last number is the number of chunks received
    _, indices = index.search(question_embedding.reshape(1, -1), 5)

    # Use the indices to retrieve the corresponding chunks
    relevant_chunks = [knowledge_base[i] for i in indices[0]]

    # Convert each chunk (which is a dictionary) into a formatted string and clean them
    cleaned_chunks = [clean_content(f"Q: {chunk['question']}\nA: {chunk['answer']}") for chunk in relevant_chunks]

    # Include the cleaned chunks in the final prompt, following the template
    prompt_content = "\n".join(cleaned_chunks) + f"\n{user_question}"
    final_prompt = SYSTEM_CONTEXT + prompt_content
    
    # Append the behavior instruction and the user's message to the conversation history
    if len(past_messages) == 0 or (len(past_messages) > 0 and past_messages[0]['content'] != SYSTEM_CONTEXT.strip()):
        past_messages.insert(0, {"role": "system", "content": SYSTEM_CONTEXT.strip()})

    past_messages.append({"role": "user", "content": prompt_content})

    # Calculate the number of tokens in the conversation history
    total_tokens = sum([tiktoken_len(message["role"] + message["content"]) for message in past_messages])

    # If the conversation history is too long, remove messages from the beginning
    while total_tokens > config['token_limit']:
        removed_message = past_messages.pop(0)
        total_tokens -= tiktoken_len(removed_message["role"] + removed_message["content"])

    # Generate the model's response
    response = openai.ChatCompletion.create(
        model=config['openai_model'], 
        messages=past_messages
    )

    generated_answer = response['choices'][0]['message']['content']

    # Append the assistant's message to the conversation history
    past_messages.append({"role": "assistant", "content": generated_answer})

    return generated_answer.strip(), past_messages

def main():
  
    knowledge_base = load_knowledge_base(config['knowledge_base_file'])
    if knowledge_base is None:
        print("Could not load the knowledge base. Exiting...")
        return
    # separates out questions and answers for embedding
    questions = [item["question"] for item in knowledge_base]
    answers = [item["answer"] for item in knowledge_base]
    
    # transformer model used to create embeddings
    model = SentenceTransformer(config['sentence_transformer_model'])

    # can load embeddings from an existing file instead of making new ones every time (for speed, assuming Q&A file doesn't change that much)
    question_embeddings = load_or_create_embeddings(config['question_embeddings_file'], questions, model)
    
    # same thing, load index instead of creating a new one every time
    index = load_or_create_index(config['index_file'], question_embeddings)

    # keeps conversation history up to a token limit
    past_messages = []
    while True:
        
        # questions derived from user input
        user_question = input("I'm Cohley's customer support assistant. What can I help you with today? (type 'exit' to stop): ").strip()
        if not user_question:
            print("Input cannot be empty, please enter a valid question.")
            continue
        if user_question.lower() == 'exit':
            break

        generated_answer, past_messages = ask_gpt(user_question, past_messages, model, index, knowledge_base)
        print(f"Answer: {generated_answer}")

if __name__ == "__main__":
    main()
