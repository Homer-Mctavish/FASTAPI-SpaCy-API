from fastapi import FastAPI
import spacy
from pydantic import BaseModel
import re
from customtokenizer import CustomSentenceTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List

nlp = spacy.load("en_core_web_sm")
app = FastAPI(tags=['sentence'])
tokenizer=CustomSentenceTokenizer()
model_name="gpt2"
gptokenizer=GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def extract_lines_from_pdf(pdf_file):
    lines = []
    # Open the PDF file
    with fitz.open(pdf_file) as pdf_document:
        # Iterate through each page
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            # Extract text from the page
            text = page.get_text()
            # Split text into lines
            lines.extend(text.split('\n'))
    return lines

class MyModel(BaseModel):
    my_list: List[str]

class Input(BaseModel):
    text: str

class UploadPDFRequest(BaseModel):
    pdf_file: UploadFile = Field(..., description="PDF file to upload")


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
async def set_delimiters(delimiter: Input):
    delimiter_list = delimiter.text.split(',')
    tokenizer.set_delimiters(delimiter_list)
    if delimiter_list != None:
        return {"message": "special characters set successfully"}
    else:
        raise HTTPException(status_code=404, detail="special characters not set")



listofpdflines=[]
@app.post("/upload-pdf/")
async def upload_pdf(pdf_file: UploadPDFRequest):
    contents = await pdf_file.read()
    gloity = extract_lines_from_pdf(contents)
    listofpdflines=gloity
    # You can now process the PDF file contents as needed
    return JSONResponse(content={"message": "File uploaded successfully"})

currentresumestring=""
def nlp_ent_detect(pdfnewlines: list):
    entlist = []
    for line in pdfnewlines:
        nlpsentence = nlp(line)
        entlist.append(nlpsentence)
    return entlist

attention_mask = []
attention_tensoro = nlp_ent_detect(listofpdflines)
# Generate attention mask based on named entity annotations
for ent in attention_tensoro:
    for j in ent.ents:
        listforent = []
        attention_mask.append(listforent)
        for i, j in range(j.start, j.end):
            attention_mask[i][j] = 1

# Convert attention mask to tensor
attention_mask_tensor = torch.tensor(attention_mask)

# @app.post("/entity_recognition")
# def get_entity(sentence_input: MyModel):
#     output_array = []
#     tokenstart= tokenizer.tokenize(sanitized)
#     texto=" ".join(tokenstart)
#     document = nlp(texto)
#     for token in document.ents:
#         output = {
#             "Text": token.text, "Start Char": token.start_char,
#             "End Char": token.end_char, "Label": token.label_
#         }
#         output_array.append(output)
#     return {"output": output_array}

def create_masking(nlpmodel):
    # Initialize attention mask
    attention_mask = [0] * len(nlpmodel)


    # Generate attention mask based on named entity annotations
    for ent in nlpmodel.ents:
        for i in range(ent.start, ent.end):
            attention_mask[i] = 1
    attention_mask_tensor = torch.tensor(attention_mask)
    return attention_mask_tensor

for stringu in separated_strings:
    # Tokenize the string using the GPT-2 tokenizer
    gptokenizer.encode(stringu, return_tensors="pt")

@app.post("/gpttext")
def training(nermask: list):
    attention_mask = create_masking(nlp)
    model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=500)
    generated_template_sentences= gptokenizer.decode(output[0], skip_special_tokens=True)

