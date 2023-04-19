import torch.nn as nn
from transformers import GPT2LMHeadModel

class GPT2DecoderWithImageFeatures(nn.Module):
    def __init__(self, input_size, gpt_model_name='gpt2'):
        super(GPT2DecoderWithImageFeatures, self).__init__()

        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        self.image_feature_projection = nn.Linear(input_size, self.gpt.config.n_embd)

    def forward(self, input, image_features):
        # Transform the image features to the GPT embedding size
        transformed_image_features = self.image_feature_projection(image_features)

        # Get the input token embeddings
        input_emb = self.gpt.transformer.wte(input)

        # Concatenate the transformed image features with the input token embeddings
        input_emb = torch.cat([transformed_image_features.unsqueeze(1), input_emb], dim=1)

        # Run the GPT model with the concatenated input
        output = self.gpt(inputs_embeds=input_emb)["logits"]

        return output
