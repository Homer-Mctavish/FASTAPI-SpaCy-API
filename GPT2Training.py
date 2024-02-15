import torch
from customtokenizer import CustomSentenceTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Sample input template_sentences
# Write template sentences with placeholders for job skills
template_sentences = [
"I believe my [SKILL] and [SKILL] skills make me a perfect candidate for your [job] role.",
"I am confident that my experience in [SKILL] and [SKILL] would be a valuable asset to your [department].",
"I am particularly interested in using my [SKILL] and [SKILL] skills to contribute to your [project].",
"I am proud to say that I have a strong background in [SKILL] and [SKILL], which I am excited to bring to your [company].",
"I am eager to learn and grow in my [SKILL] and [SKILL] abilities, and I believe your [company] would be a great place to do that.",
"I am confident that my skills in [SKILL] and [SKILL] would allow me to hit the ground running in your [department].",
"I am dedicated to using my [SKILL] and [SKILL] abilities to solve problems and drive results for your [company].",
"I am excited to apply my [SKILL] and [SKILL] skills to your [project].",
"I have a strong track record of using my [SKILL] and [SKILL] abilities to improve [metric].",
"I am particularly proud of my [SKILL] and [SKILL] skills, which have been instrumental in my [achievement].",
"I am particularly experienced in using [tool] and [tool], which I believe will be valuable to your [department].",
"I am particularly skilled in using [tool] to [task], which has allowed me to [achievement].",
"I am highly proficient in [tool], which has been instrumental in my [project].",
"14, I am confident that my ability to use [tool] and [tool] will allow me to excel in your [department].",
"I am particularly proud of my ability to [task] using [tool], which has been highly effective in [project].",
"I am particularly experienced in using [tool] to [task], which has allowed me to [achievement]."
    # Add more template sentences as needed
]


tokenizer = CustomSentenceTokenizer()
tokenizer.set_delimiters(['•','*','-','+', ':','●'])
tokenizer.default_regex_pattern()
token = tokenizer.tokenize(template_sentences)
text = " ".join(tokens)

#alternative using nltk NER over SpaCY
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# for sent in tokenlist:
#   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
#      if hasattr(chunk, 'label'):
#         print(chunk.label(), ' '.join(c[0] for c in chunk))

# Process template_sentences with SpaCy NER model
doc = nlp(text)

# Initialize attention mask
attention_mask = [0] * len(doc)

# Generate attention mask based on named entity annotations
for ent in doc.ents:
    for i in range(ent.start, ent.end):
        attention_mask[i] = 1

# Convert attention mask to tensor
attention_mask_tensor = torch.tensor(attention_mask)

# Print attention mask
print("Attention Mask:", attention_mask)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Sample input template_sentences
input_template_sentences = "The quick brown fox"

# Tokenize input template_sentences
input_ids = tokenizer.encode(input_template_sentences, return_tensors="pt")

# Generate attention mask tensor (1 for valid tokens, 0 for padding)
attention_mask = torch.ones_like(input_ids)

# Set attention mask for padding tokens to 0
attention_mask[input_ids == tokenizer.pad_token_id] = 0

# Generate template_sentences with attention mask
output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)

# Decode generated output
generated_template_sentences = tokenizer.decode(output[0], skip_special_tokens=True)

# Print generated template_sentences
print("Generated template_sentences:", generated_template_sentences)