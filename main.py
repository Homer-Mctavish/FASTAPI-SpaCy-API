from fastapi import FastAPI
import spacy
from pydantic import BaseModel
import re
from customtokenizer import CustomSentenceTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel


nlp = spacy.load("en_core_web_sm")
app = FastAPI(tags=['sentence'])
tokenizer=CustomSentenceTokenizer()
model_name="gpt2"
gptokenizer=GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

class Input(BaseModel):
    text: str

@app.put("/set_delimiters")
async def set_delimiters(delimiter: Input):
    delimiter_list = delimiter.text.split(',')
    tokenizer.set_delimiters(delimiter_list)
    if delimiter_list != None:
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



currentresumestring=""
@app.post("/entity_recognition")
def get_entity(sentence_input: Input):
    output_array = []
    tokenstart= tokenizer.tokenize(sentence_input.text)
    texto=" ".join(tokenstart)
    document = nlp(texto)
    for token in document.ents:
        output = {
            "Text": token.text, "Start Char": token.start_char,
            "End Char": token.end_char, "Label": token.label_
        }
        output_array.append(output)
    return {"output": output_array}

def create_masking(nlpmodel):
    # Initialize attention mask
    attention_mask = [0] * len(nlpmodel)


    # Generate attention mask based on named entity annotations
    for ent in nlpmodel.ents:
        for i in range(ent.start, ent.end):
            attention_mask[i] = 1
    attention_mask_tensor = torch.tensor(attention_mask)
    return attention_mask_tensor


def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

input_template_letters = load_doc('training.txt')

tokenized_text = gptokenizer.encode(input_template_letters, return_tensors="pt")

@app.post("/gpttext")
def training(nermask: list):
    attention_mask = create_masking(nlp)
    model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=500)
    generated_template_sentences= gptokenizer.decode(output[0], skip_special_tokens=True)

