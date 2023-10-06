""" this counts the number of tokens in a string or file
    to run in command line type: 
    'python token_counter.py [filename.txt or "string in quotes"]' """

import sys
import os
import tiktoken

def tiktoken_len(text):
    """
    Counts the number of tokens in a string
    """
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

def read_file(filename):
    """
    This function reads a file and returns its content as a string so that the number of tokens can be counted
    """
    try:
        with open(filename, 'r') as file:
            data = file.read()
        return data
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
        return None

def main(input_arg):
    """
    Main function to read a text file or process a string and print its token count.
    """
    if os.path.isfile(input_arg):
        content = read_file(input_arg)
        if content is None:
            print("Could not read the file. Exiting...")
            return
    else:
        content = input_arg

    token_count = tiktoken_len(content)
    print(f"The token count is: {token_count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide a file path or a string as an argument.")
        sys.exit(1)
    
    main(sys.argv[1])
