# app.py
import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import torch
from transformers import GPT2Tokenizer
from PIL import Image
from model import GPT2DecoderWithImageFeatures
from transformers import ViTModel
import numpy as np
from utils import special_tokens_dict
from generate import generate_code
import requests

app = Flask(__name__)

# Load the tokenizer, ViT model, and decoder
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens(special_tokens_dict)
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').base_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)

decoder = GPT2DecoderWithImageFeatures(input_size=768)
decoder.gpt.resize_token_embeddings(len(tokenizer))  # Update the GPT2 model with the new tokenizer

decoder.load_state_dict(torch.load("best_decoder.pth",map_location=device))
decoder.to(device)

# Add this method to save the received image
def save_image(image_data):
    img_data = base64.b64decode(image_data)
    img = Image.open(BytesIO(img_data))
    image_path = "received_image.png"
    img.save(image_path)
    return image_path

@app.route('/generate_code', methods=['POST'])
def generate_code_endpoint():
    api_key = request.json['api_key']
    image_data = request.json['image_data']
    image_path = save_image(image_data)
    generated_code = generate_code(image_path, tokenizer, vit_model, decoder)

    response_text = call_gpt3(api_key, generated_code)
    return jsonify({"generated_html": response_text})

def call_gpt3(api_key, generated_code):
    system_prompt = "You are a helpful assistant."
    chat_gpt_request = get_configured_chat_gpt_request([
        {
            "role": "system",
            "content": system_prompt
        },
        { "role": "user", "content": '''
        
        '''},
    ])

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=chat_gpt_request, headers=headers)
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as error:
        print(f"Error: {error}")
        return None

def get_configured_chat_gpt_request(messages):
    return {
        "messages": messages,
        "max_tokens": 3000,
        "model": "gpt-4",
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080), debug=True)
