from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# Define a function to generate text with custom attention mask
# Define a function to generate text with custom attention mask
def generate_with_attention_mask(input_text, attention_mask=None, max_length=500, repetition_penalty=2.0, temperature=0.8, top_p=0.9):
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Create default attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids).to(device)

    # Generate text with the model, using the input text as context
    output = model.generate(input_ids,
                             attention_mask=attention_mask,
                             max_length=max_length,
                             do_sample=True,
                             temperature=temperature,
                             top_p=top_p,
                             repetition_penalty=repetition_penalty,
                             num_return_sequences=1
                            )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# Example usage
input_text = "Dear Hiring Manager, I am excited to apply for the Cloud Site and Reliability Engineer position at Beta Technologies, and I believe my expertise in Amazon Web Services (AWS) will make me a valuable asset to your team. As an AWS Certified Cloud Practitioner, I have a deep understanding of AWS services, including EC2, S3, Code pipeline, SMS, Lambda, and many more. My experience working with AWS has enabled me to implement scalable and cost-effective solutions for various projects."
attention_mask_in = torch.tensor([[1]]).to(device)  # Example of attention mask, assuming all tokens are attended to
max_length = 1000  # Adjust the maximum length as needed

generated_cover_letter = generate_with_attention_mask(input_text, attention_mask=None, max_length=max_length)
print(generated_cover_letter)

