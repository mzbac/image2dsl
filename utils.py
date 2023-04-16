from torchvision import transforms
import re

vocabulary = ', { } small-title text quadruple row btn-inactive btn-orange btn-green btn-red double <START> header btn-active <END> single <UNK> <PAD>'.split()

vocabulary.append('')

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
        pattern = r'[ ,\n]+'
        tokens = re.split(pattern, text)        
        encoded_tokens = []
        for token in tokens:
            if token in self.vocab:
                encoded_tokens.append(self.vocab.index(token))
            else:
                print(f"Unknown vocabulary: '{token}'")
                encoded_tokens.append(self.vocab.index("<UNK>"))
        return encoded_tokens

    def decode(self, tokens):
        return ' '.join([self.inv_vocab[token] for token in tokens])

tokenizer = CustomTokenizer(vocabulary)