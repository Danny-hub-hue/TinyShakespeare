import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from modelTrainingParams import GRUModel, ShakespeareDataset, CharTokenizer



with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print(tokenizer.texts_to_sequences("Hello World"))

model = GRUModel(max_id=65)

def generate_text(seed_text, model_path, max_length=100, temperature=1.0):
    model.load_state_dict(torch.load(model_path, map_location = torch.device("cpu")))
    model.eval()

    generated = seed_text
    for _ in range(max_length):
        encoded = tokenizer.texts_to_sequences(generated[-100:])
        encoded = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(encoded)

        # Apply temperature to logits and sample from the distribution
        predictions = predictions[:, -1, :] / temperature
        probabilities = torch.nn.functional.softmax(predictions, dim=-1)
        predicted_index = torch.multinomial(probabilities, 1).item()

        generated += tokenizer.index_to_char[predicted_index]

    return generated

#Declare GPU and move model to it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running prediciton on: {device}")
model.to(device)

# Example usage
#model_path = 'tiny_shakespeare_model.pkl'
#seed_text = "Hello "
#generated_text = generate_text(seed_text, model_path, max_length=100)
#print(generated_text)