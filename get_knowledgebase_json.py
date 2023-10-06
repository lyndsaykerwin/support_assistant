"""
This script fetches the knowledge base from Freshdesk (specifically the 'Creator Support' category) and stores it as a JSON file so it can be used to answer customer support questions.
"""

import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Constants
DOMAIN = 'cohley.freshdesk.com'
FAQ_LOCATION = 'Creator Support'
CREATOR_SUPPORT_CATEGORY_ID = 48000424440
API_KEY = os.getenv('FRESHDESK_API_KEY')
PASSWORD = 'x'  # This is not used but is expected by the Freshdesk API

def get_folders_in_category(category_id):
    """
    Gets all of the folders inside a given category.
    category_id : int
        The ID of the category to fetch folders from.
    Returns : list
        A list of all articles in the category.
    """
    url = f"https://{DOMAIN}/api/v2/solutions/categories/{category_id}/folders"
    response = requests.get(url, auth=(API_KEY, PASSWORD))

    if response.status_code == 200:
        data = response.json()
        all_articles = []
        for folder in data:
            all_articles.extend(get_articles_in_folder(folder['id']))
        return all_articles
    else:
        raise Exception(f"Failed to get folders in category {category_id}. Status code: {response.status_code}")

def get_articles_in_folder(folder_id):
    """
    Pulls all of the articles from the folders inside a given category.
    folder_id : int
        The ID of the folder to fetch articles from.
    Returns : list
        A list of articles in the folder, each represented as a dictionary.
    """
    url = f"https://{DOMAIN}/api/v2/solutions/folders/{folder_id}/articles"
    response = requests.get(url, auth=(API_KEY, PASSWORD))

    if response.status_code == 200:
        data = response.json()
        folder_articles = [{'question': article['title'], 'answer': article['description']} for article in data]
        return folder_articles
    else:
        raise Exception(f"Failed to get articles in folder {folder_id}. Status code: {response.status_code}")

def fetch_knowledge_base(category_id=CREATOR_SUPPORT_CATEGORY_ID):
    """
    Fetches knowledge base for a specific category.
    category_id : int
        The ID of the category to fetch knowledge base for.
    Returns : list
        A list of all articles in the category.
    """
    knowledge_base = get_folders_in_category(category_id)

    # Save the knowledge_base to a json file
    with open('knowledgebase.json', 'w') as json_file:
        json.dump(knowledge_base, json_file)

    print("Knowledgebase saved to 'knowledgebase.json'")
    return knowledge_base

# If the script is run as the main module
if __name__ == "__main__":
    fetch_knowledge_base()
