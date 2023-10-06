# customer support assistant

This project contains a customer support assistant built on OpenAI GPT. It uses a combination of SentenceTransformer embeddings, FAISS for search, and OpenAI's GPT for natural language understanding and generation. GPT answers customer support questions based on user input, supplied with relevant info from Cohley's Freshdesk knowledgebase.

## Features

- Pulls knowledgebase info from Freshdesk API calls.
- Searches the knowledgebase for the chunks of information relevant to a customer support question using SentenceTransformer and FAISS.
- Uses GPT to generate responses to user queries
- Tracks and manages conversation history to stay within a token limit set in a configuration file

## Other things to know

- Loading the Transformer model makes the program start-up run slow, but provides a way for us to manage the number of tokens given to GPT to answer a question
- my understanding from Erik is that it may be okay if this runs slow 
- I've tried a couple other models that are supposed to be faster but my computer can't handle them. I'm not sure if there's another way to fix this issue

## Requirements

- Python 3.x
- SentenceTransformer
- FAISS
- OpenAI API & API key
- Freshdesk API & API key

## Installation

```bash
pip install -r requirements.txt
