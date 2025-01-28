from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch
import json
import os

# Load the tokenizer and model (T5-base for manageable performance/memory)
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Set the model path and optimizer path
model_path = "./scryfall_translator_t5_large"
optimizer_path = "optimizer.pt"

# Flag to skip training if model already exists
skip_training = True  # Change to False if you want to always retrain

# Try loading the saved model and optimizer if they exist
if os.path.exists(model_path) and skip_training:
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    print("Loaded saved model.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Recreate optimizer
    try:
        optimizer.load_state_dict(torch.load(optimizer_path))
        print("Loaded saved optimizer state.")
    except:
        print("No saved optimizer state found.")
else:
    print("No saved model found or skip_training is False, starting with a new model.")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Custom Dataset class for the training data
class ScryfallDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data[idx][0]
        target_text = self.data[idx][1]
        return input_text, target_text

# Load the training data from the JSON file
file_path = "training_data.json"
with open(file_path, "r") as f:
    training_data = json.load(f)

# Prepare the dataset and DataLoader
train_dataset = ScryfallDataset(training_data)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Use batch size of 16

# Fine-tuning loop with batches if skip_training is False
if not skip_training:
    # Set the model to training mode
    model.train()
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            inputs_batch = [item[0] for item in batch]
            labels_batch = [item[1] for item in batch]
            
            # Tokenize batch inputs and outputs
            inputs = tokenizer(inputs_batch, return_tensors="pt", padding=True, truncation=True).to(device)
            labels = tokenizer(labels_batch, return_tensors="pt", padding=True, truncation=True).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels.input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss}")

    # Save the fine-tuned model and optimizer state
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    torch.save(optimizer.state_dict(), optimizer_path)

# Function to generate Scryfall queries with refined post-processing
def translate_to_scryfall_query(input_text):
    model.eval()
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Rule-based fixes for common query patterns:
    
    # Mono-color handling
    if "mono-black" in input_text.lower():
        query = query.replace("color black", "color:b")
    elif "mono-red" in input_text.lower():
        query = query.replace("color red", "color:r")
    elif "mono-green" in input_text.lower():
        query = query.replace("color green", "color:g")
    elif "mono-blue" in input_text.lower():
        query = query.replace("color blue", "color:u")
    elif "mono-white" in input_text.lower():
        query = query.replace("color white", "color:w")
    
    # Handling tutors
    if "tutor" in input_text.lower():
        query = query.replace("o:tutors", "o:tutor")
    
    # Return the final query
    return query

# Test the model
while True: 
    test_query = input('MTGPT: ')
    print("Generated Scryfall Query:", translate_to_scryfall_query(test_query))
