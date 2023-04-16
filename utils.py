import torch
from torchvision import transforms

vocabulary = ', { } small-title text quadruple row btn-inactive btn-orange btn-green btn-red double <START> header btn-active <END> single <UNK> <PAD>'.split()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {i: token for i, token in enumerate(vocab)}

    def encode(self, text):
        tokens = text.split()
        return [self.vocab.index(token) if token in self.vocab else self.vocab.index("<UNK>") for token in tokens]

    def decode(self, tokens):
        return ' '.join([self.inv_vocab[token] for token in tokens])

tokenizer = CustomTokenizer(vocabulary)
