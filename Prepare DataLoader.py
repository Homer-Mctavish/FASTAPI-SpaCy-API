# Prepare DataLoader
dataset = CustomDataset(tokenized_text)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(0))[0]
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the trained model
model.save_pretrained("trained_model")

# Generate template_sentences with attention mask
output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1300)

# Decode generated output
generated_template_sentences = tokenizer.decode(output[0], skip_special_tokens=True)

