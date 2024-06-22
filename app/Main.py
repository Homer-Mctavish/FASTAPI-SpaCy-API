from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
from transformers import pipeline
from pydantic import BaseModel
from typing import Optional

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

# Load Llama 3 model from Hugging Face
llama3_model = pipeline("text-generation", model="MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF")

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


# Generate text using the Llama 3 model
prompt = "Once upon a time"
generated_text = llama3_model(prompt, max_length=50, do_sample=True)


class PromptRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 50
    do_sample: Optional[bool] = True

@app.post("/generate-text/")
async def generate_text(request: PromptRequest):
    try:
        generated_text = llama3_model(request.prompt, 250, request.do_sample)
        return {"generated_text": generated_text[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Print the generated text
print(generated_text[0]['generated_text'])


