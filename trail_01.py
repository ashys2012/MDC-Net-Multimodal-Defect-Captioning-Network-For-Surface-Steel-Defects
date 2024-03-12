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
from train_val_epoch import train_epoch, valid_epoch_bbox
from allied_files import CFG, concat_gt
from iou_bbox import iou_loss_individual



txt_file_path = "/mnt/sdb/2024/pix_2_seq_with_captions_march/annotations_summary_fin.txt"
image_folder = "/mnt/sdb/2024/pix_2_seq_with_captions_march/images"

df = txt_file_to_df(txt_file_path, image_folder)
df['img_path'] = df['img_path'].apply(lambda x: f"{x}.jpg" if not x.lower().endswith('.jpg') else x)
df = df[~df['ids'].isin(['patches_211.jpg', 'scratches_211.jpg'])]

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
print("The df is of length:", len(df))
print("All image paths in the DataFrame exist.")

# Check for a specific file existence
file_path = '/kaggle/input/neu-steel-surface-defect-detect-trainvalid-split/valid_images/patches_283.jpg'
if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")
else:
    print(f"The file '{file_path}' does not exist.")


# Apply concat_gt to include bounding boxes (captions are handled separately)
df['concatenated'] = df.apply(concat_gt, axis=1)

# # Group by 'ids' and aggregate data
df = df.groupby('ids').agg({
    'concatenated': lambda x: list(x),  # Aggregates bounding boxes into a list per 'ids'
    'img_path': 'first',  # Assumes img_path is the same for all entries with the same 'ids'
    'caption': 'first'  # Captures the single caption associated with each 'ids'
}).reset_index()

df['concatenated'] = df.apply(lambda row: row['concatenated'], axis=1)



all_captions = df['caption'].tolist() 



vocab = Vocabulary(freq_threshold=5)
vocab.build_vocab(all_captions) 
tokenizer = Tokenizer(vocab, num_classes=6, num_bins=CFG.num_bins,
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





# Usage
train_loader, valid_loader = get_loaders(
    df, tokenizer, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code
)



# for i, batch in enumerate(valid_loader):
#     if batch is None:
#         continue
#     images, labels, bboxes = batch
#     print(f"Batch {i+1}")
#     print("Images:", images.shape)
#     print("Labels:", labels)
#     print("BBoxes:", bboxes)
#     break



encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=512)
decoder = Decoder(vocab_size=305, #complete_vocab_size, #tokenizer.vocab_size,
                  encoder_length=CFG.num_patches, dim=512, num_heads=16, num_layers=12)
model = EncoderDecoder(encoder, decoder)

model.to(CFG.device)


        

def train_eval(model, train_loader, valid_loader, criterion, optimizer, lr_scheduler, step, logger):
    best_metric = float('inf')  # Adjust based on whether lower or higher is better for your primary metric
    epochs_since_improvement = 0
    patience = CFG.patience

    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch + 1}")

        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler if step == 'batch' else None, criterion, logger=logger)

        # Validation phase with correct arguments
        # Note: Make sure to only call valid_epoch_bbox once with the correct set of arguments
        valid_loss, avg_iou, total_loss = valid_epoch_bbox(model, valid_loader, criterion, tokenizer, iou_loss_individual, CFG)

        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}, Avg IoU: {avg_iou:.3f}, Total Loss: {total_loss:.3f}")

        # Example: Check for improvement based on average IoU
        if avg_iou < best_metric:  # Assuming lower is better; adjust the comparison operator as needed
            best_metric = avg_iou
            epochs_since_improvement = 0  # Reset counter
            save_path = 'output_path/best_model_by_iou_1.pth'
            torch.save(model.state_dict(), save_path)
            print("Saved Model with Best Average IoU")
        else:
            epochs_since_improvement += 1

        if lr_scheduler is not None and step == 'epoch':
            lr_scheduler.step()

        # Early stopping check
        if epochs_since_improvement >= patience:
            print("Early stopping triggered after no improvement for", patience, "epochs.")
            break

        # Optionally log other metrics or details here



optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

num_training_steps = CFG.epochs * (len(train_loader.dataset) // CFG.batch_size)
num_warmup_steps = int(0.1 * num_training_steps)
lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                               num_training_steps=num_training_steps,
                                               num_warmup_steps=num_warmup_steps)

criterion = nn.CrossEntropyLoss(ignore_index=CFG.pad_idx) #alpha=1, gamma=2, 

train_eval(model,
           train_loader,
           valid_loader,
           criterion,
           optimizer,
           lr_scheduler=lr_scheduler,
           step='batch',
           logger=None)

