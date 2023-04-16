import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class CustomTransformerDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CustomTransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.image_feature_projection = nn.Linear(input_size, hidden_size)
        self.transformer_decoder_layer = TransformerDecoderLayer(hidden_size, nhead=8, dim_feedforward=hidden_size*4)
        self.transformer_decoder = TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, image_features):
        embedded = self.embedding(input)
        embedded = embedded.permute(1, 0, 2)
        image_features = self.image_feature_projection(image_features)
        image_features = image_features.unsqueeze(0).repeat(embedded.size(0), 1, 1)
        output = self.transformer_decoder(embedded, image_features)
        output = self.fc(output)
        return output

