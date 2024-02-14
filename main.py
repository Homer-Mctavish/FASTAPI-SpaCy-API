from fastapi import FastAPI
import spacy
from pydantic import BaseModel
import re

en_core_web = spacy.load("en_core_web_sm")

app = FastAPI(tags=['sentence'])

class CustomSentenceTokenizer:
    def __init__(self):
        self.regex_pattern = r'(?<=[.!?])\s+'
        self.delimiters = ['*', '-', '+']
    
    def set_regex_pattern(self, regex_pattern):
        self.regex_pattern = regex_pattern
    
    def set_delimiters(self, delimiters):
        self.delimiters = delimiters
    
    def default_regex_pattern(self):
        self.regex_pattern = r'(?<=[.!?])\s+'
    
    def tokenize(self, paragraph):
        # Define the pattern to split the paragraph into sentences
        pattern = self.regex_pattern
        for delimiter in self.delimiters:
            pattern += f"|{re.escape(delimiter)}"  # Escaping special characters
        # Split the paragraph into sentences using the pattern
        sentences = re.split(pattern, paragraph)
        # Remove empty sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences

# Example usage:
tokenizer = CustomSentenceTokenizer()

# Set custom regex pattern
tokenizer.set_regex_pattern(r'\n+')

# Set custom delimiters
tokenizer.set_delimiters(['*', '-', '+'])


class Input(BaseModel):
    sentence: str

@app.post("/set_delimiters")
def set_delimiters(delimiter: Input):
    delimiter_list = delimiter.list
    tokenizer.set_delimiters(delimiter_list)
    listofdelim=[]
    for delimiter in delimiter_list:
        delim = {"char": delimiter}
        listofdelim.append(delim)
    return {"delimiters": listofdelim}

@app.post("/set_regex")
def set_regex(regexpress: Input):
    regular = regexpress.regex
    tokenizer.set_regex_pattern(regular)
    return {"regular expression": regular}

@app.post("/set_delimiters")
def set_delimiters(delimiter: Input):
    delimiter_list = delimiter.list
    tokenizer.set_delimiters(delimiter_list)
    listofdelim = {"delimiters": delimiter_list}
    return listofdelim

@app.post("/analyze_text")
def get_text_characteristics(sentence_input: Input):
    document = en_core_web(sentence_input.sentence)
    output_array = []
    for token in document:
        output = {
            "Index": token.i, "Token": token.text, "Tag": token.tag_, "POS": token.pos_,
            "Dependency": token.dep_, "Lemma": token.lemma_, "Shape": token.shape_,
            "Alpha": token.is_alpha, "Is Stop Word": token.is_stop
        }
        output_array.append(output)
    return {"output": output_array}

@app.post("/entity_recognition")
def get_entity(sentence_input: Input):
    document = en_core_web(sentence_input.sentence)
    output_array = []
    for token in document.ents:
        output = {
            "Text": token.text, "Start Char": token.start_char,
            "End Char": token.end_char, "Label": token.label_
        }
        output_array.append(output)
    return {"output": output_array}



