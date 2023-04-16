import torch
from PIL import Image
import numpy as np
from transformers import ViTModel
from utils import vocabulary, tokenizer, img_transform
from model import CustomTransformerDecoder

def generate_code(decoder, img_path, vit_model, tokenizer, img_transform, decoder_weights_path, vit_weights_path, max_len=128):
 # Set the models to evaluation mode
    decoder.eval()
    vit_model.eval()
    
    device = torch.device("cpu")

    # Load saved model weights
    saved_decoder_weights = torch.load(decoder_weights_path, map_location=device)

    # Load the weights into the models
    decoder.load_state_dict(saved_decoder_weights)

    # Load and preprocess image
    img_rgb = Image.open(img_path)
    img_grey = img_rgb.convert("L")
    img_adapted = img_grey.point(lambda x: 255 if x > 128 else 0)
    img_stacked = np.stack((img_adapted, img_adapted, img_adapted), axis=-1)
    img_stacked_pil = Image.fromarray(np.uint8(img_stacked), mode='RGB')
    img_tensor = img_transform(img_stacked_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = vit_model(img_tensor).last_hidden_state[:, 0, :]

    # Initialize input sequence with <START> token
    input_sequence = [tokenizer.encode('<START>')[0]]
    input_sequence = torch.LongTensor(input_sequence).unsqueeze(1).to(device)

    generated_tokens = []

    # Generate tokens one-by-one
    for _ in range(max_len):
        with torch.no_grad():
            output = decoder(input_sequence, image_features)

        # Get the token with the highest probability
        token = output[-1, 0].argmax().item()

        # Print the generated token
        print(f"Generated token: {token} - {tokenizer.decode([token])}")

        # Stop generating tokens when <END> token is produced
        if token == tokenizer.encode('<END>')[0]:
            break

        generated_tokens.append(token)
        input_sequence = torch.cat([input_sequence, torch.LongTensor([token]).unsqueeze(1)], dim=1)

    # Convert generated tokens to code
    generated_code = tokenizer.decode(generated_tokens)
    return generated_code

if __name__ == "__main__":
    img_path = "images.png"
    decoder_weights_path = "best_decoder.pth"
    vit_weights_path = "fine_tuned_vit_model.pth"

    # Instantiate the ViT model and the decoder
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').base_model
    input_size = 768
    hidden_size = 256
    output_size = len(vocabulary)
    num_layers = 3
    decoder = CustomTransformerDecoder(input_size, hidden_size, output_size, num_layers)

    generated_code = generate_code(decoder, img_path, vit_model, tokenizer, img_transform, decoder_weights_path, vit_weights_path)
    print(generated_code)
