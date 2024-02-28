from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
from pydantic import BaseModel
import re
from customtokenizer import CustomSentenceTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List
import torch
import fitz  # PyMuPDF


nlp = spacy.load("en_core_web_sm")
app = FastAPI(tags=['sentence'])
tokenizer=CustomSentenceTokenizer()
model_name="gpt2"
gptokenizer=GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
currentresumestring=""

# Define CORS settings
origins = [
    "http://localhost",
    "http://localhost:5173",  # Add your specific origin here
    # Add more origins as needed
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Add allowed HTTP methods
    allow_headers=["*"],  # Allow any headers, you can customize this as needed
)

class StringInput(BaseModel):
    longString: str

def nlp_ent_detect(pdfnewlines: list):
    entlist = []
    for line in pdfnewlines:
        nlpsentence = nlp(line)
        entlist.append(nlpsentence)
    return entlist


# Open the text file for reading
with open('training.txt', 'r') as file:
    # Read the content of the file
    lines = file.readlines()

# Initialize lists to store separated strings
separated_strings = []
current_string = []

# Iterate through the lines of the file
for line in lines:
    # Check if the line contains "Dear Hiring Manager,"
    if "Dear Hiring Manager," in line:
        # If there's a current string, append it to the list of separated strings
        if current_string:
            separated_strings.append(''.join(current_string))
            current_string = []  # Reset current string
    # Append the current line to the current string
    current_string.append(line)

# Append the last current string to the list of separated strings
if current_string:
    separated_strings.append(''.join(current_string))

@app.put("/set_delimiters")
async def set_delimiters(delimiters: StringInput):
    tokenizer.set_delimiters(delimiters.split(','))
    if delimiter_list != None:
        return {"message": "special characters set successfully"}
    else:
        raise HTTPException(status_code=404, detail="special characters not set")

long_string_length = 1
entities=[]
@app.post("/endpoint")
async def receive_string(string_input: StringInput):
    output_array = []
    long_string = string_input.longString
    long_string_length = len(string_input.longString)
    listofpdflines = long_string.splitlines()
    tokenstart= tokenizer.tokenize(long_string)
    texto=" ".join(tokenstart)
    document = nlp(texto)
    entities = [(ent.start_char, ent.end_char) for ent in document.ents]
    for token in document.ents:
        output = {
            "Text": token.text, "Start Char": token.start_char,
            "End Char": token.end_char, "Label": token.label_
        }
        output_array.append(output)
    return {"output": output_array}

@app.post("/lelsos")
async def ert():
    return{"Mass age": entities}

# Create attention mask
attention_mask = [0] * long_string_length
for start, end in entities:
    attention_mask[start:end] = [1] * (end - start)

# Pad attention mask to match input sequence length
max_seq_length = 500
attention_mask += [0] * (max_seq_length - len(attention_mask))
attention_mask = attention_mask[:max_seq_length]

# Convert attention mask to tensor
attention_mask_tensor = torch.tensor(attention_mask)

for stringu in separated_strings:
    # Tokenize the string using the GPT-2 tokenizer
    gptokenizer.encode(stringu, return_tensors="pt")

@app.post("/gpttext")
def training():
    text = model.generate(input_ids=input_ids, attention_mask=attention_mask_tensor, max_length=500)
    generated_template_sentences= gptokenizer.decode(output[0], skip_special_tokens=True)
    if text != None:
        return {"message": text}
    else:
        raise HTTPException(status_code=404, detail="failed to generate letter")


