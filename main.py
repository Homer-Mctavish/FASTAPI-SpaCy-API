from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
from pydantic import BaseModel
import re
from customtokenizer import CustomSentenceTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from typing import List
import torch
import fitz  # PyMuPDF

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = ("cpu")

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


# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

#torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
torch.set_default_tensor_type(torch.FloatTensor)
model.to(device)


# Prepare dataset
dataset = TextDataset(tokenizer=tokenizer, file_path="training.txt", block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Create Trainer instance and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

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

def createmask(entitylistfromspacy: list, long_string: int):
    # Create attention mask
    attention_mask = [0] * long_string
    for start, end in entitylistfromspacy:
        attention_mask[start:end] = [1] * (end - start)

    # Pad attention mask to match input sequence length
    max_seq_length = 500
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    attention_mask = attention_mask[:max_seq_length]

    # Convert attention mask to tensor
    attention_mask_tensor = torch.tensor(attention_mask).to(device)
    return attention_mask_tensor


@app.post("/gpttext")
def training(input_text: str):
    if entities:
        model.to(device)
        attention_mask_set = createmask(entites, long_string_length)
        input_ids = gptokenizer.encode(input_text, return_tensors="pt").to(device)
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask_set, max_length=500)
        generated_template_coverletter= gptokenizer.decode(output[0], skip_special_tokens=True)
    else:
        model.to(device)
        input_ids = gptokenizer.encode(input_text, return_tensors="pt").to(device)
        output = model.generate(input_ids=input_ids, max_length=500)
        generated_template_coverletter = gptokenizer.decode(output[0], skip_special_tokens=True)
    if generated_template_coverletter:
        return {"message": generated_template_coverletter}
    else:
        raise HTTPException(status_code=404, detail="failed to generate letter")


