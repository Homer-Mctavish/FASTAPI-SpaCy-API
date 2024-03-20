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

# Print the separated strings
for i, string in enumerate(separated_strings, start=1):
    print(f"String {i}:")
    print(string.strip())
    print()


for stringu in separated_strings:
    # Tokenize the string using the GPT-2 tokenizer
    gptokenizer.encode(stringu, return_tensors="pt")