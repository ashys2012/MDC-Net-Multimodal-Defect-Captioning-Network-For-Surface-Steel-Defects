import torch
import random
import os
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class CFG:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    max_len = 100
    img_size = 224
    num_bins = img_size

    debug = False
    
    batch_size = 64
    epochs = 150
    
    model_name = 'deit3_medium_patch16_224.fb_in22k_ft_in1k'
    num_patches = 196
    lr = 1e-5     # reduced to 5 from 4
    weight_decay = 1e-4       # changed (reduced)frfom 1e-5 to counter overfitting

    generation_steps = 101
    l1_lambda = 1e-7             #kept high as the model is complex
    patience = 40
    iou_loss_weight = 0.5
    model_save_path = "/mnt/sdb/2024/pix_2_seq_with_captions_GC_10_dataset/output_1"


#below is from the original
# def generate_square_subsequent_mask(sz):
#     mask = (torch.triu(torch.ones((sz, sz), device=CFG.device))
#             == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float(
#         '-inf')).masked_fill(mask == 1, float(0.0))
#     return mask

#here we convert the function to make it float 32
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz), device=CFG.device), diagonal=1)
    return mask.float()  # Convert to float32



#below is from the original
# def create_mask(tgt):
#     """
#     tgt: shape(N, L)
#     """
#     tgt_seq_len = tgt.shape[1]

#     tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
#     tgt_padding_mask = (tgt == CFG.pad_idx)

#     return tgt_mask, tgt_padding_mask

#we convert this to float 32 as well
def create_mask(tgt):
    tgt_seq_len = tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_padding_mask = (tgt == CFG.pad_idx).float()  # Convert to float32
    return tgt_mask, tgt_padding_mask



class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0]*3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

def concat_gt(row):
    # Extract necessary information from the row
    label = row['label']
    xmin = row['xmin']
    xmax = row['xmax']
    ymin = row['ymin']
    ymax = row['ymax']
    # Note: We're not including the caption here to avoid redundancy
    return [label, xmin, ymin, xmax, ymax]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]