import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from PIL import Image
import numpy as np
from model import GPT2DecoderWithImageFeatures
from utils import img_transform, special_tokens_dict

# Load and preprocess data
class Pix2CodeDataset(Dataset):
    def __init__(self, data_path, img_transform, dsl_transform, mode="train", split_ratio=0.8):
        self.data_path = data_path
        self.img_transform = img_transform
        self.dsl_transform = dsl_transform
        self.mode = mode
        self.split_ratio = split_ratio
        self.data = self.load_data()

    def load_data(self):
        data = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    dsl_path = os.path.splitext(img_path)[0] + ".gui"
                    data.append((img_path, dsl_path))

        split_index = int(len(data) * self.split_ratio)
        if self.mode == "train":
            return data[:split_index]
        elif self.mode == "val":
            return data[split_index:]
        else:
            raise ValueError("Invalid mode. Use 'train' or 'val'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, dsl_path = self.data[idx]
        img_rgb = Image.open(img_path)
        img_grey = img_rgb.convert("L")
        img_adapted = img_grey.point(lambda x: 255 if x > 128 else 0)
        img_stacked = np.stack((img_adapted, img_adapted, img_adapted), axis=-1)
        img_stacked_pil = Image.fromarray(np.uint8(img_stacked), mode='RGB')

        with open(dsl_path, "r") as f:
            dsl_code = f.read()

        img_tensor = self.img_transform(img_stacked_pil)
        dsl_tokens = self.dsl_transform('<START>\n' + dsl_code + '\n<END>')
        dsl_tokens = self.dsl_transform('<START>\n' + dsl_code + '\n<END>')
        dsl_tensor = torch.LongTensor(dsl_tokens)

        return img_tensor, dsl_tensor

# Initialize ViT model, dataset, and data loader
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').base_model

# Replace the dsl_transform with the tokenizer.encode method
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens(special_tokens_dict)
dsl_transform = tokenizer.encode

# Create train and validation data loaders
data_path = "data/all_data/data"

train_dataset = Pix2CodeDataset(data_path, img_transform, dsl_transform, mode="train")
val_dataset = Pix2CodeDataset(data_path, img_transform, dsl_transform, mode="val")

def pad_collate_fn(batch):
    imgs, dsls = zip(*batch)

    # Pad DSL sequences
    max_len = max([len(dsl) for dsl in dsls])
    end_token = tokenizer.encode('<PAD>')[0]
    padded_dsls = []
    for dsl in dsls:
        padded_dsls.append(torch.cat([dsl, torch.full((max_len - len(dsl),), end_token,dtype=torch.long)]))

    # Stack padded DSL sequences and images
    img_tensor = torch.stack(imgs)
    dsl_tensor = torch.stack(padded_dsls)

    return img_tensor, dsl_tensor

batch_size = 32
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

# Define hyperparameters
input_size = 768
num_layers = 6
epochs = 1000
learning_rate = 0.0001

# Initialize the decoder, loss function, and optimizer
decoder = GPT2DecoderWithImageFeatures(input_size)
decoder.gpt.resize_token_embeddings(len(tokenizer))  # Update the GPT2 model with the new tokenizer
decoder.load_state_dict(torch.load("best_decoder.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(decoder.parameters(), lr=learning_rate)

# Initialize the learning rate scheduler with warm-up
num_warmup_steps = 500
num_training_steps = epochs * len(train_data_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

# Initialize variables to track the best validation loss and epoch
best_val_loss = float("inf")
best_epoch = 0

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)
decoder.to(device)

vit_model.eval()

for epoch in range(epochs):
    for i, (img_tensor, dsl_tensor) in enumerate(train_data_loader):
        loss = 0
        decoder.train()
        img_tensor = img_tensor.to(device)
        dsl_tensor = dsl_tensor.to(device)

        # Extract image features using ViT model
        with torch.no_grad():
            image_features = vit_model(img_tensor).last_hidden_state[:, 0, :]

        # Training loop for the GPT2DecoderWithImageFeatures
        input_tokens = dsl_tensor[:, :-1]
        target_tokens = dsl_tensor[:, 1:]
       
        output = decoder(input_tokens, image_features)

        loss = criterion(output.permute(0, 2, 1), target_tokens)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=2.0)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        scheduler.step()

        if i % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Step {i}/{len(train_data_loader)}, Loss: {loss.item()}")

    # Evaluate the model on the validation set
    val_loss = 0
    decoder.eval()
    with torch.no_grad():
        for i, (img_tensor, dsl_tensor) in enumerate(val_data_loader):
            img_tensor = img_tensor.to(device)
            dsl_tensor = dsl_tensor.to(device)

            image_features = vit_model(img_tensor).last_hidden_state[:, 0, :]

            input_tokens = dsl_tensor[:, :-1]
            target_tokens = dsl_tensor[:, 1:]     

            output = decoder(input_tokens, image_features)

            val_loss += criterion(output.permute(0, 2, 1), target_tokens).item()

    val_loss /= len(val_data_loader)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")
    # Save the best model weights
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(decoder.state_dict(), "best_decoder.pth")

print(f"Best model weights saved from epoch {best_epoch+1} with validation loss {best_val_loss}")
