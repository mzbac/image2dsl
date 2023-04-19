import torch
from transformers import GPT2Tokenizer
from PIL import Image
from model import GPT2DecoderWithImageFeatures
from transformers import ViTModel
from torchvision import transforms
import numpy as np
from utils import img_transform, special_tokens_dict


def generate_code(image_path, tokenizer, vit_model, decoder, max_length=512,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Load and preprocess the image
    img_rgb = Image.open(image_path)
    img_grey = img_rgb.convert("L")
    img_adapted = img_grey.point(lambda x: 255 if x > 128 else 0)
    img_stacked = np.stack((img_adapted, img_adapted, img_adapted), axis=-1)
    img_stacked_pil = Image.fromarray(np.uint8(img_stacked), mode='RGB')
    img_tensor = img_transform(img_stacked_pil)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Extract image features using ViT model
    with torch.no_grad():
        image_features = vit_model(img_tensor).last_hidden_state[:, 0, :]

    # Prepare the initial input for the decoder
    input_ids = tokenizer.encode('<START>', return_tensors='pt').to(device)

    # Generate code using the decoder
    decoder.eval()
    generated_code = []
    for _ in range(max_length):
        with torch.no_grad():
            output = decoder(input_ids, image_features)

        next_token_id = torch.argmax(output, dim=-1)[:, -1]
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

        if next_token_id.item() == tokenizer.encode('<END>')[0]:
            break

        # Replace the button color with 'btn' to address the bias in the training dataset
        button_colors = ['btn-orange', 'btn-green', 'btn-red']

        if tokenizer.decode(next_token_id) in button_colors:
            generated_code.append('btn')
        else:
            generated_code.append(tokenizer.decode(next_token_id))

    return ''.join(generated_code)


if __name__ == "__main__":
    # Load the tokenizer, ViT model, and decoder
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens(special_tokens_dict)
    vit_model = ViTModel.from_pretrained(
        'google/vit-base-patch16-224').base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model.to(device)

    decoder = GPT2DecoderWithImageFeatures(input_size=768)
    # Update the GPT2 model with the new tokenizer
    decoder.gpt.resize_token_embeddings(len(tokenizer))

    decoder.load_state_dict(torch.load(
        "best_decoder.pth", map_location=device))
    decoder.to(device)

    # Perform inference on a test image
    image_path = "images.png"
    generated_code = generate_code(image_path, tokenizer, vit_model, decoder)
    print("Generated code:\n", generated_code)
