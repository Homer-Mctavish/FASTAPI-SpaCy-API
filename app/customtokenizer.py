import re

class CustomSentenceTokenizer:
    def __init__(self):
        self.regex_pattern = r'(?<=[.!?])\s+'
        self.delimiters = ['*', '-', '+']
    
    def set_regex_pattern(self, regex_pattern):
        self.regex_pattern = regex_pattern
    
    def set_delimiters(self, delimiters):
        self.delimiters = delimiters
    
    def default_regex_pattern(self):
        self.regex_pattern = r'(?<=[.!?])\s+'
    
    def tokenize(self, paragraph):
        # Define the pattern to split the paragraph into sentences
        pattern = self.regex_pattern
        for delimiter in self.delimiters:
            pattern += f"|{re.escape(delimiter)}"  # Escaping special characters
        # Split the paragraph into sentences using the pattern
        sentences = re.split(pattern, paragraph)
        # Remove empty sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences

