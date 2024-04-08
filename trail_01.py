import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dataset import txt_file_to_df, get_transform_train, get_transform_valid, collate_fn
from allied_files import seed_everything, CFG
from model import Encoder, Decoder, EncoderDecoder
import torch
import cv2
from functools import partial
from sklearn.model_selection import train_test_split
from utils import AvgMeter, get_lr
from tqdm import tqdm
from torch import nn
import transformers
from transformers import top_k_top_p_filtering
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
seed_everything(seed=42)
#import wandb
#wandb.init(project="pix_2_seq")

from data_processing import Tokenizer, Vocabulary, get_loaders
from allied_files import CFG, concat_gt
from iou_bbox import iou_loss_individual
from train_val_epoch import train_epoch,valid_epoch_bbox, test_epoch
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CyclicLR



from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn


txt_file_path = "/mnt/sdb/2024/pix_2_seq_with_captions_GC_10_dataset/annotations_summary.txt"
image_folder = "/mnt/sdb/2024/pix_2_seq_with_captions_GC_10_dataset/data/images"

df = txt_file_to_df(txt_file_path, image_folder)
df['img_path'] = df['img_path'].apply(lambda x: f"{x}.jpg" if not x.lower().endswith('.jpg') else x)
#df = df[~df['ids'].isin(['patches_211.jpg', 'scratches_211.jpg'])]

# Validate and Filter Invalid Image Paths
existing_img_paths = df['img_path'].apply(lambda x: os.path.exists(x))
df = df[existing_img_paths].reset_index(drop=True)

# Validate Bounding Boxes (assuming bbox data is in the DataFrame)
# This part is hypothetical and depends on your DataFrame structure
# bboxes_validity, message = check_bbox_validity(df['bboxes'])
# if not bboxes_validity:
#     print(message)
# else:
#     print("All bboxes are valid.")

print("The device is:", CFG.device)
print("The df is of length before concatenated:", len(df))
print("All image paths in the DataFrame exist.")

# Check for a specific file existence
file_path = '/kaggle/input/neu-steel-surface-defect-detect-trainvalid-split/valid_images/patches_283.jpg'
if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")
else:
    print(f"The file '{file_path}' does not exist.")


# # Apply concat_gt to include bounding boxes (captions are handled separately)
# df['concatenated'] = df.apply(concat_gt, axis=1)

# print("THe original df length is before feeding into the get_loaders is and after concatenated is:", len(df))

# df = df.groupby('img_path').agg({
#     'concatenated': lambda x: list(x),  # Aggregates bounding boxes into a list per image
#     'ids': 'first',  # Captures the single label associated with each image
#     'caption': 'first'  # Assumes caption is the same for all entries with the same image
# }).reset_index()

# print(df.head())


# print("THe original df length is before feeding into the get_loaders is and after df.groupby is:", len(df))

# df['concatenated'] = df.apply(lambda row: row['concatenated'], axis=1)



all_captions = df['caption'].tolist() 



vocab = Vocabulary(freq_threshold=5)
vocab.build_vocab(all_captions) 
tokenizer = Tokenizer(vocab, num_classes=10, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
CFG.bos_idx = tokenizer.BOS_code
CFG.pad_idx = tokenizer.PAD_code



tokenizer.print_vocab()


# Get Vocabulary Sizes
spacy_vocab_size = len(tokenizer.vocab)
print(f"Spacy Vocabulary size is {spacy_vocab_size}")

complete_vocab_size = tokenizer.vocab_size
print(f"Complete Vocabulary size is {complete_vocab_size}")

total_vocab_size = spacy_vocab_size + complete_vocab_size
print("total_vocab_size is ", total_vocab_size)   



sampled_df = df.sample(frac=0.1, random_state=42)


# Usage
train_loader, valid_loader, test_loader = get_loaders(
    df, tokenizer, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code
)

"""
Here starts WANDB implementation
"""

import wandb

#Initialize wandb and pass configuration from CFG class

wandb.init(project="pix_2_seq_march_224_gc_10_ds", entity="ashys2012", name = "server_v_large_1024_8_8_patch_dr_002_iou_03_cyclic_lr_total_vocab_size_correct_num_cls", config={
    "device": CFG.device.type,  # Logging the device type as a string
    "max_len": CFG.max_len,
    "img_size": CFG.img_size,
    "num_bins": CFG.num_bins,
    "batch_size": CFG.batch_size,
    "epochs": CFG.epochs,
    "model_name": CFG.model_name,
    "num_patches": CFG.num_patches,
    "lr": CFG.lr,
    "weight_decay": CFG.weight_decay,
    "generation_steps": CFG.generation_steps,
    "l1_lambda": CFG.l1_lambda,
    "patience": CFG.patience,
    "iou_loss_weight": CFG.iou_loss_weight,
    "model_save_path": CFG.model_save_path,
})


logger = wandb

#logger = None

encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=1024)
decoder = Decoder(vocab_size=total_vocab_size, #complete_vocab_size, #tokenizer.vocab_size,
                  encoder_length=CFG.num_patches, dim=1024, num_heads=8, num_layers=8)
model = EncoderDecoder(encoder, decoder)

model.to(CFG.device)


# Assuming CFG and other necessary imports are already defined

def train_eval(model, train_loader, valid_loader, criterion, tokenizer, optimizer, lr_scheduler, step, logger=None):
    best_metric = float('inf')  # Adjust based on whether lower or higher is better for your primary metric
    epochs_since_improvement = 0
    patience = CFG.patience

    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch + 1}/{CFG.epochs}")

        # Training phase
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler if step == 'batch' else None, criterion, logger = logger, iou_loss_weight=CFG.iou_loss_weight)

        # Validation phase
        model.eval()
        valid_loss, avg_giou, total_loss = valid_epoch_bbox(model, valid_loader, criterion, tokenizer, iou_loss_weight=CFG.iou_loss_weight, logger=logger, epoch_num=epoch)
        
        test_epoch(model, test_loader, tokenizer, save_dir='/mnt/sdb/2024/pix_2_seq_with_captions_GC_10_dataset/test_output_images',logger=logger, epoch_num=epoch)
        # Update the learning rate based on warmup scheduler if step is 'epoch'
        if lr_scheduler is not None and step == 'epoch':
            lr_scheduler.step()

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {valid_loss:.4f}, Avg GIoU: {avg_giou:.4f}, Total Loss: {total_loss:.4f}")

        # Improvement check based on the metric of interest (e.g., avg_giou here)
        if avg_giou < best_metric:  # Adjust based on your metric; here lower avg_giou indicates improvement
            best_metric = avg_giou
            epochs_since_improvement = 0
            save_path = f'output_1/best_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Saved Improved Model at {save_path}")
        else:
            epochs_since_improvement += 1

        # Early stopping
        if epochs_since_improvement >= patience:
            print("Early stopping triggered. No improvement in Avg GIoU for", patience, "epochs.")
            break



optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

num_training_steps = CFG.epochs * (len(train_loader.dataset) // CFG.batch_size)
print("num_training_steps is:", num_training_steps)
num_warmup_steps = int(0.075 * num_training_steps)
print("num_warmup_steps is:", num_warmup_steps)
# lr_scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                num_training_steps=num_training_steps,
#                                                num_warmup_steps=num_warmup_steps)

# Example parameters - adjust based on your dataset and model complexity
base_lr = 1e-7  # Minimum learning rate
max_lr = 1e-4   # Maximum learning rate
step_size_up = len(train_loader) // 2  # Half an epoch to ramp up

lr_scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular', cycle_momentum=False)
# lr_scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                num_training_steps=num_training_steps,
#                                                num_warmup_steps=num_warmup_steps)

criterion = nn.CrossEntropyLoss(ignore_index=CFG.pad_idx) #alpha=1, gamma=2, 

train_eval(model,
           train_loader,
           valid_loader,
           criterion,
           optimizer = optimizer,
           tokenizer=tokenizer,
           lr_scheduler=lr_scheduler,
           step='batch',
           logger=logger)
