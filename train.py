import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
# from transformers import BertTokenizer, BertModel
from CLIP import clip
from utils.create_phoc_label import *
from utils.evaluate import QbE, QbS
from config.options import opts
from fastkan import FastKAN as KAN
from fastkan import AttentionWithFastKANTransform as AttKAN
from fastkan import TransformerEncoderWithKAN, multi_AttentionWithFastKAN
import numpy as np
import cv2 as cv
import math
import glob
from open_clip import create_model_from_pretrained, create_model_and_transforms
# from pretrain_clip import model as pretrainclip
# Freeze model utility functions
def freeze_model(m):
    m.requires_grad_(False)

# def freeze_all_but_bn(m):
#     for name, param in m.named_parameters():
#         if 'ln' not in name:
#             param.requires_grad = False

# def freeze_all_but_ln(m):
#     """冻结所有层，除了 LayerNorm 层"""
#     for name, param in m.named_parameters():
#         if 'ln' not in name.lower():  # 兼容 CLIP 的 LayerNorm 命名方式
#             param.requires_grad = False

# def freeze_all_but_bn(m):
#     if not isinstance(m, torch.nn.LayerNorm):
#         if hasattr(m, 'weight') and m.weight is not None:
#             m.weight.requires_grad_(False)
#         if hasattr(m, 'bias') and m.bias is not None:
#             m.bias.requires_grad_(False)
            
# def freeze_all_but_bn(model):
#     for name, module in model.named_modules():
#         if not isinstance(module, torch.nn.LayerNorm):
#             for param in module.parameters():
#                 param.requires_grad = False
#         else:            
#             for param in module.parameters():
#                 param.requires_grad = True
#                 # print(name, module,param.requires_grad)
                
def all_train(model):
    for name, module in model.named_modules():                   
        for param in module.parameters():
            param.requires_grad = True
                
def cosine_distance(x, y):
    return 1.0 - F.cosine_similarity(x, y)

def mse_distance(x, y):
    # 计算均方误差
    return torch.mean((x - y) ** 2, dim=-1)

    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params

def delete_digit(word_label):
    # 剔除1、2
    for label in word_label:
        if label.isdigit():
            word_label = word_label.replace(label, '')
    return word_label

class CrossModalTranslator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024):
        super().__init__()        
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.kan1 = KAN([input_dim, hidden_dim])
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.kan2 = KAN([hidden_dim, input_dim])
        self.attkan = AttKAN(out_dim=512, q_dim=512, k_dim=512, v_dim=512, num_heads=4)
        self.norm = nn.LayerNorm(input_dim)  # 归一化
        self.dropout = nn.Dropout(0.1)  # 轻微正则化
        

    def forward(self, x):
        identity = x  # 残差连接
        out = self.kan1(x)
        out = self.relu(out)
        out = self.kan2(out)
        out = self.attkan(out, out, out)
        out = self.dropout(out)
        out = self.norm(out + identity)  # 残差连接 + 归一化
        return out
    
class multi_PHOC(nn.Module):
    def __init__(self, opts):
        super(multi_PHOC, self).__init__()
        hs = 772 if opts.exp_name.split('_')[0] == 'Kanjur' else 604
        self.fc1 = nn.Linear(512, hs)
        self.fc2 = nn.Linear(hs, 512)
    def forward(self, x, phoc):
        return self.fc2(self.fc1(x) * phoc)                        
         
class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.opts = opts
        self.device = device
        # self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip, _, _ = create_model_and_transforms(model_name='ViT-B-32-quickgelu', pretrained="pretrained_weights/CLIP/MetaCLIP/b32_fullcc2.5b.pt")
        # self.clip.apply(freeze_all_but_ln)
        
        self.multi_PHOC = multi_PHOC(self.opts)
        # 跨模态转换器
        self.translator2text = CrossModalTranslator()        
        self.translator2img = CrossModalTranslator()
        
        
        self.TripletLoss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=0.2)
        # self.img_text_align_loss = nn.MSELoss()
        self.align_loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        
    def forward(self, data, type='img', convert=False, prompt=None, phoc=None):
        
        if type == 'img':
            if prompt is not None:
                feat = self.clip.encode_image(data, prompt.expand(data.shape[0], -1, -1))
            else:
                feat = self.clip.encode_image(data)
            if convert:  # 图像转换为文本
                feat = self.translator2text(feat)
        elif type == 'text':
            data = torch.cat([clip.tokenize(f"a word image of a {delete_digit(c)}") for c in data]).to(self.device)
            feat = self.clip.encode_text(data)
            feat = self.multi_PHOC(feat, phoc)
            if convert:  # 文本转换为图像
                feat = self.translator2img(feat)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat   

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    total_triplet_loss = 0
    total_align_retri_loss = 0
    total_align_phoc_loss = 0
        
    for batch in tqdm(train_loader, desc="Training"):
        anchor_img_tensor, pos_img_tensor, neg_img_tensor, word, neg_word, phoc = batch[:6]
        anchor_img_tensor, pos_img_tensor, neg_img_tensor = anchor_img_tensor.to(device), pos_img_tensor.to(device), neg_img_tensor.to(device)        
        phoc = phoc.to(device).to(torch.float32)
        optimizer.zero_grad()
        
        anc_feat = model(anchor_img_tensor, 'img')
        pos_feat = model(pos_img_tensor, 'img')
        neg_feat = model(neg_img_tensor, 'img')        
        loss_triplet_img = model.TripletLoss(anc_feat, pos_feat, neg_feat)
        
        text_feat = model(word, 'text', phoc=phoc)
        # neg_text_feat = model(neg_word, 'text', phoc=phoc)
        # loss_triplet_text = model.TripletLoss(text_feat, text_feat, neg_text_feat)
        # loss_ali_img_text = model.align_loss(anc_feat, text_feat)
        
        img2text = model(anchor_img_tensor, type='img', convert=True)
        text2img = model(word, type='text', convert=True, phoc=phoc)
        
        loss_ali_img = model.align_loss(anc_feat, text2img)
        loss_ali_text = model.align_loss(text_feat, img2text)
    
        loss = loss_triplet_img + loss_ali_img + loss_ali_text
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Compute average losses
    avg_total_loss = total_loss / len(train_loader)

    # Print epoch average losses
    print(f"Total: {avg_total_loss:.4f}")
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    all_img_fea, all_text_fea, all_category = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            anchor_img_tensor, word, phoc= batch[:3]
            anchor_img_tensor = anchor_img_tensor.to(device)
            phoc = phoc.to(device).to(torch.float32)
            
            img_feat = model(anchor_img_tensor, type='img')
            
            # 所有文本特征通过转换器转换为图像特征
            txt_feat = model(word, type='text', convert=True, phoc=phoc) 
            
            all_img_fea.append(img_feat.cpu())
            all_text_fea.append(txt_feat.cpu())
            all_category.append(word)
    
    all_img_fea = torch.cat(all_img_fea)
    all_text_fea = torch.cat(all_text_fea)
    all_category = np.array(sum([list(cat) for cat in all_category], []))
    
    query_index = [i for i in range(all_category.shape[0])]
    qbe = QbE(model.opts, all_category, all_img_fea, 0, query_index, fold=None, drop_first=True)
    qbs, _ = QbS(model.opts, all_text_fea, all_img_fea, all_category, all_category, 0, query_index)    
    return qbe, qbs


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(filename='checkpoint.pth'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        return None
    
if __name__ == '__main__':
    if opts.exp_name.split('_')[0] == 'Geser':
        from dataset.Geser import Geser as DATA
    elif opts.exp_name.split('_')[0] == 'IAM':
        from dataset.IAM import IAM as DATA
    elif opts.exp_name.split('_')[0] == 'Kanjur':
        from dataset.Kanjur import Kanjur as DATA
        
    dataset_transforms = DATA.data_transform(opts)
    train_dataset = DATA(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = DATA(opts, dataset_transforms, mode='val', return_orig=False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(device).to(device)
    
    total_params, trainable_params, frozen_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Frozen parameters: {frozen_params}")   
    
    # # Wrap model with DataParallel if multiple GPUs are available
    # print("Number of available GPUs:", torch.cuda.device_count())
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
        
    optimizer = torch.optim.Adam([
    {'params': model.clip.parameters(), 'lr': opts.clip_LN_lr},
    {'params': model.translator2text.parameters(), 'lr': 1e-4},
    {'params': model.translator2img.parameters(), 'lr': 1e-4},
    {'params': model.multi_PHOC.parameters(), 'lr': 1e-4},
    
])
    
    # 加载检查点
    checkpoint = load_checkpoint(f'saved_models/{opts.exp_name}/last_ckpt.pth')
    start_epoch = 0
    best_qbe = -1e3
    best_qbs = -1e3

    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_qbe = checkpoint['best_qbe']
        best_qbs = checkpoint['best_qbs']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")

    num_epochs = 2000
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        qbe, qbs = validate(model, val_loader, device)
        
        with open('save_results/{}_mAP.txt'.format(opts.exp_name), 'a') as f:
            f.write('epoch {}: QBE, {}; QBS, {}\n'.format(epoch, qbe, qbs))
        print(opts.exp_name)
        print(f'Epoch {epoch}: QBE mAP: {qbe:.4f}, QBS mAP: {qbs:.4f}')
                
        os.makedirs(f'saved_models/{opts.exp_name}', exist_ok=True)
        # Update best QBE and QBS scores
        if qbe > best_qbe:
            best_qbe = qbe
            old_files = glob.glob(f'saved_models/{opts.exp_name}/best_qbe_epoch_*.pth')
            if old_files:
                os.remove(old_files[0])  # 删除第一个匹配的文件
            # Save the model with the best QBE score
            torch.save(model.state_dict(), f'saved_models/{opts.exp_name}/best_qbe_epoch_{epoch}.pth')
        
        if qbs > best_qbs:
            best_qbs = qbs
            old_files = glob.glob(f'saved_models/{opts.exp_name}/best_qbs_epoch_*.pth')
            if old_files:
                os.remove(old_files[0])  # 删除第一个匹配的文件
            # Save the model with the best QBS score
            torch.save(model.state_dict(), f'saved_models/{opts.exp_name}/best_qbs_epoch_{epoch}.pth')
            
        save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_qbe': best_qbe,
                        'best_qbs': best_qbs,
                    }, filename=f'saved_models/{opts.exp_name}/last_ckpt.pth')
         # Print best scores
        print(f'Best QBE mAP: {best_qbe:.4f}, Best QBS mAP: {best_qbs:.4f}\n')

