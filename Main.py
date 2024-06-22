from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
from pydantic import BaseModel
from typing import Optional
import re
from customtokenizer import CustomSentenceTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
from typing import List
import torch
from spacy.pipeline import EntityRuler
from spacy.language import Language
import json
import os
import replicate

# Retrieve the Hugging Face token from the environment variable
rp_token = os.getenv("REPLICATE_API_TOKEN")

if not rp_token:
    raise ValueError("Replicate token not found. Please set the REPLICATE_API_TOKEN environment variable.")


app = FastAPI(tags=['sentence'])

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


def create_entity_ruler(nlp, name):
    # Create a new EntityRuler
    ruler = EntityRuler(nlp, name=name)

    # Load training data from JSONL file
    with open("skill_patterns.jsonl", "r") as file:
        for line in file:
            data = json.loads(line)
            text = data["pattern"]
            label = data["label"]
            # Create pattern dictionary
            pattern = {"label": label, "pattern": text}
            # Add pattern to the EntityRuler
            ruler.add_patterns([pattern])

    return ruler

Language.factory("ent_rule", func=create_entity_ruler)


# Load existing spaCy model from disk
nlp = spacy.load("en_core_web_sm")

class StringInput(BaseModel):
    longString: str


# Add the EntityRuler to the pipeline and specify a name
nlp.add_pipe('ent_rule', name="entity_ruler2", before="ner")



@app.post("/endpoint")
async def receive_string(input_string: StringInput):
    tokenized_string = tokenizer.tokenize(input_string.longString)
    text = " ".join(tokenized_string)
    doc = nlp(text)
    entities = [(start, end) for start, end in [(ent.start_char, ent.end_char) for ent in doc.ents]]
    response_array = [
        {"text": entity.text, "start_char": entity.start_char,
         "end_char": entity.end_char, "label": entity.label_} for entity in doc.ents
    ]
    return {"output": response_array}




class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate-text/")
async def generate_text(request: PromptRequest):
    try:
        input = {
            "top_p": 0.9,
            "prompt": request.prompt,
            "min_tokens": 0,
            "temperature": 0.6,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15
        }

        for event in replicate.stream(
            "meta/meta-llama-3-70b-instruct",
            input=input
        ):
            print(event, end="")
            return({"generated_text": event})
#=> "Let's break this problem down step by step.\n\nStep 1: S...
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

