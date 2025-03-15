import torch
from open_clip import create_model_from_pretrained, create_model_and_transforms, get_tokenizer, tokenize
from CLIP import clip


def load_clip_model_OpenAICLIP(device):
    
    model, _ = clip.load('ViT-B/32', device=device)
    
    return model.to(torch.float32)


# def load_clip_model_DFN(config):

#     class_model = DFN(config)
#     class_model.to(torch.float32)

#     return class_model


def load_clip_model_SigLIP(device):

    model, _ = create_model_from_pretrained(model_name='ViT-SO400M-14-SigLIP', pretrained="pretrained_weights/CLIP/SigLIP/ViT-SO400M-14-SigLIP/open_clip_pytorch_model.bin",
                                                    image_mean=([0.5,0.5,0.5]), image_std=([0.5,0.5,0.5]), image_interpolation="bicubic", image_resize_mode="squash")
    # tokenizer = get_tokenizer('pretrained_weights/CLIP/SigLIP/ViT-SO400M-14-SigLIP')
    return model.to(torch.float32)


def load_clip_model_MetaCLIP(device):
    
    model, _, _ = create_model_and_transforms(model_name='ViT-B-32-quickgelu', pretrained="pretrained_weights/CLIP/MetaCLIP/b32_fullcc2.5b.pt")

    return model.to(torch.float32)