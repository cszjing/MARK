import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import cv2 as cv
from .create_phoc_label import unigrams_from_new_historical_mongolian_word_list, get_most_common_n_grams, build_phoc
import json

def get_phoc(path, mode='train'):
    with open(os.path.join(path, 'trainset.txt'), 'r')as f:
        lines = f.readlines()
    train_word = [line.strip().split('\\')[0] for line in lines]
    with open(os.path.join(path, 'testset.txt'), 'r')as f:
        lines = f.readlines()
    test_word = [line.strip().split('\\')[0] for line in lines]            
    all_word = train_word + test_word
    # phoc_unigrams
    if os.path.exists(os.path.join(path, 'phoc_unigrams.txt')):
        phoc_unigrams = np.loadtxt(os.path.join(path, 'phoc_unigrams.txt'), delimiter=',', dtype=str)
    else:
        phoc_unigrams = unigrams_from_new_historical_mongolian_word_list(word_label=all_word)
        np.savetxt(os.path.join(path, 'phoc_unigrams.txt'), phoc_unigrams, delimiter=',', fmt='%s')
    # bigrams
    if os.path.exists(os.path.join(path, 'bigrams.txt')):
        with open(os.path.join(path, 'bigrams.txt'), 'r') as file:
            bigrams = json.loads(file.read())
    else:
        bigrams = get_most_common_n_grams(words=[word for word in all_word],
                                    num_results=50, n=2)
        with open(os.path.join(path, 'bigrams.txt'), 'w') as file:
            file.write(json.dumps(bigrams, indent=4))
            
    if mode == 'train':       
        phoc_label = build_phoc(words=[word for word in train_word],
                                phoc_unigrams=phoc_unigrams, unigram_levels=[elem for elem in range(2, 6)],
                                phoc_bigrams=bigrams, bigram_levels=[2], on_unknown_unigram='warn')
    else:
        phoc_label = build_phoc(words=[word for word in test_word],
                                phoc_unigrams=phoc_unigrams, unigram_levels=[elem for elem in range(2, 6)],
                                phoc_bigrams=bigrams, bigram_levels=[2], on_unknown_unigram='warn')
    return phoc_label


class Geser(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig
        # self.all_words = os.listdir(self.opts.data_dir)
        self.set_path = os.path.join(self.opts.data_split_dir, 'oov', 'split_rate_{}'.format(self.opts.data_split)) 
        self.mode = mode       
        if self.mode == 'train':            
            self.set_dir = os.path.join(self.set_path, 'trainset.txt')
        else:
            self.set_dir = os.path.join(self.set_path, 'testset.txt')
        with open(self.set_dir, 'r')as f:
            lines = f.readlines()
        self.all_set_path = [os.path.join(self.opts.data_dir, '/'.join(line.strip().split('\\')))  for line in lines]
        self.all_train_word = list(set([line.strip().split('\\')[0] for line in lines]))        
        
        self.phoc_label = get_phoc(self.set_path, self.mode)
   

 
    def __len__(self):
        return len(self.all_set_path)
        
    def __getitem__(self, index):
        filepath = self.all_set_path[index] 
        phoc = self.phoc_label[index]
                   
        word = filepath.split(os.path.sep)[-2]        
        neg_words = self.all_train_word.copy()
        neg_words.remove(word)
        neg_word = np.random.choice(neg_words)
        
        anchor_path = filepath
        pos_path = os.path.join(self.opts.data_dir, word, np.random.choice(os.listdir(os.path.join(self.opts.data_dir, word))))        
        neg_path = os.path.join(self.opts.data_dir, neg_word, np.random.choice(os.listdir(os.path.join(self.opts.data_dir, neg_word))))
        
        
        anchor_img = ImageOps.pad(Image.open(anchor_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        pos_img = ImageOps.pad(Image.open(pos_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_img = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))    
        
        anchor_img_tensor = self.transform(anchor_img) # (224, 224, 3) ---> torch.Size([3, 224, 224])
        pos_img_tensor = self.transform(pos_img)
        neg_img_tensor = self.transform(neg_img)
        
        
        if self.mode == 'train':
            return (anchor_img_tensor, pos_img_tensor, neg_img_tensor, word, neg_word, phoc)
        else:
            return (anchor_img_tensor, word, phoc)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms


