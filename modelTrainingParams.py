import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import requests
from tqdm import tqdm


class GRUModel(nn.Module):
    def __init__(self, max_id):
        super(GRUModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=max_id, embedding_dim=max_id)
        self.gru = nn.GRU(input_size=max_id, hidden_size=1024, num_layers=2, batch_first=True, dropout=0.05)
        self.fc = nn.Linear(1024, max_id)
    
    def forward(self, x):
        x = self.embed(x)
        output, _ = self.gru(x)
        output = self.fc(output)
        return output


class ShakespeareDataset(Dataset):
    def __init__(self, encoded_text, sequence_length=100):
        self.data = encoded_text
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, index):
        return (torch.tensor(self.data[index:index+self.sequence_length], dtype=torch.long),
                torch.tensor(self.data[index+1:index+self.sequence_length+1], dtype=torch.long))



class CharTokenizer:
    def __init__(self):
        self.char_to_index = {}
        self.index_to_char = {}
    
    def fit_on_texts(self, texts):
        unique_chars = set(texts)
        self.char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
    
    def texts_to_sequences(self, texts):
        return [self.char_to_index[char] for char in texts]