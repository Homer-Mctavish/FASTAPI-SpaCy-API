from fastapi import FastAPI
import spacy
from pydantic import BaseModel
import re
from customtokenizer import CustomSentenceTokenizer

en_core_web = spacy.load("en_core_web_sm")

app = FastAPI(tags=['sentence'])

tokenizer=CustomSentenceTokenizer()

class Input(BaseModel):
    sentence: str


@app.put("/set_delimiters")
async def set_delimiters(delimiter: Input):
    delimiter_list = delimiter.sentence.split(',')
    tokenizer.set_delimiters(delimiter_list)
    if listofdelim != None:
        return {"message": "special characters set successfully"}
    else:
        raise HTTPException(status_code=404, detail="special characters not set")



@app.put("/set_regex")
def set_regex(regexpress: Input):
    try:
        regular = regexpress.text
        tokenizer.set_regex_pattern(regular)
        return {"message": "regular expression set successfully"}
    except:
        raise HTTPException(status_code=404, detail="special characters not set")


# @app.post("/analyze_text")
# def get_text_characteristics(sentence_input: Input):
#     document = en_core_web(sentence_input.sentence)
#     output_array = []
#     for token in document:
#         output = {
#             "Index": token.i, "Token": token.text, "Tag": token.tag_, "POS": token.pos_,
#             "Dependency": token.dep_, "Lemma": token.lemma_, "Shape": token.shape_,
#             "Alpha": token.is_alpha, "Is Stop Word": token.is_stop
#         }
#         output_array.append(output)
#     return {"output": output_array}

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

def create_masking(data):
    # Initialize attention mask
    attention_mask = [0] * len(doc)


    # Generate attention mask based on named entity annotations
    for ent in doc.text:
        for i in range(ent.start, ent.end):
            attention_mask[i] = 1

# Tokenize input template_sentences
input_ids = tokenizer.encode(input_template_sentences, return_tensors="pt")

# Generate template_sentences with attention mask
output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)

# Decode generated output
generated_template_sentences = tokenizer.decode(output[0], skip_special_tokens=True)

model_name="gpt2"
gptokenizer=GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
@app.post("/gptraining")
def training(nermask: list):
    model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=500)
    generated_template_sentences= gptokenizer.decode(output[0], skip_special_tokens=True)

