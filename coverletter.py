from transformers import pipeline

def generate_cover_letter(job_description):
    # Load the text generation pipeline
    text_generator = pipeline("text-generation", model="GPT-2-finetuned-common_gen")
    
    # Define the cover letter template
    cover_letter_template = f"""
    [Your Name]
    [Your Address]
    [City, State, Zip Code]
    [Your Email Address]
    [Your Phone Number]
    [Today’s Date]

    [Hiring Manager's Name]
    [Company Name]
    [Company Address]
    [City, State, Zip Code]

    Dear [Hiring Manager's Name],

    I am writing to express my interest in the [Job Title] position at [Company Name]. With a background in [relevant field or skills mentioned in the job description], I am excited about the opportunity to contribute to your team and help [specific goal or project mentioned in the job description].

    {job_description}

    One aspect of the [Company Name] that particularly appeals to me is [mention something specific about the company]. I am eager to join a team of passionate individuals who share my enthusiasm for [mention any specific interests or goals mentioned in the job description].

    I am confident that my combination of skills, experiences, and passion make me a strong fit for the [Job Title] position at [Company Name]. I am enthusiastic about the opportunity to further discuss how my background, skills, and passions align with the needs of your team. Thank you for considering my application. I look forward to the possibility of contributing to [Company Name] and am available at your earliest convenience for an interview.

    Warm regards,

    [Your Name] 
    """
    
    # Generate the cover letter
    cover_letter = text_generator(cover_letter_template, max_length=500)[0]['generated_text']
    
    return cover_letter

# Example usage
job_description = """
                Job Description: We are seeking a skilled Python developer with experience in natural language processing (NLP) using the Hugging Face Transformers library. The ideal candidate should have a strong background in machine learning and be proficient in Python, TensorFlow, and PyTorch. Responsibilities include developing and deploying NLP models, fine-tuning pre-trained models, and collaborating with cross-functional teams to drive innovation in NLP.
                """

generated_cover_letter = generate_cover_letter(job_description)
print(generated_cover_letter)
