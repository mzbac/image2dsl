from torchvision import transforms

vocabulary = ', { } small-title text quadruple row btn-inactive btn-orange btn-green btn-red double <START> header btn-active <END> single <UNK> <PAD>'.split()
special_tokens_dict = {'additional_special_tokens': vocabulary}

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])