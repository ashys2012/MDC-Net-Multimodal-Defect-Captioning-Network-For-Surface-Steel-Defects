import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from functools import partial
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from functools import partial
import albumentations as A
import pandas as pd
import cv2
import albumentations as A
import torch
from transformers import top_k_top_p_filtering
import matplotlib.pyplot as plt
import matplotlib.patches as patches  
from allied_files import CFG, seed_everything, concat_gt
from dataset import txt_file_to_df, get_transform_train, get_transform_valid, collate_fn
from data_processing import Tokenizer, Vocabulary
from model import Encoder, Decoder, EncoderDecoder
import torch
from PIL import Image
from torchvision import transforms
from data_processing import top_k_sampling
import torch
from tqdm import tqdm

max_ln = 100
vocab_size = 305

def decode_sequence(sequence, tokenizer):
    """
    Decodes a sequence of tokens into human-readable form, including captions and bounding boxes.
    Each token in the sequence is converted to a tensor before being passed to the tokenizer's decode method.
    """
    decoded = []
    for token in sequence:
        # Convert each integer token to a tensor and specify the device
        token_tensor = torch.tensor([token], device=CFG.device)
        # Decode the tensor to get the corresponding string representation
        decoded_token = tokenizer.decode(token_tensor)
        decoded.append(decoded_token)
    return decoded


def inference_single_image(model, image_tensor, tokenizer, top_k=5):
    model.eval()
    with torch.no_grad():
        x = image_tensor.to(CFG.device)
        y_input = torch.tensor([[tokenizer.BOS_code]], device=CFG.device)
        seq_generated = []

        for _ in range(max_ln):  # max_ln is the maximum sequence length
            preds = model(x, y_input)
            logits = preds[:, -1, :]  # Get the logits for the last token position
            next_token = top_k_sampling(logits, k=top_k)  # Use top-k sampling
            seq_generated.append(next_token.item())
            
            if next_token.item() == tokenizer.EOS_code:
                break
            
            y_input = torch.cat([y_input, next_token], dim=-1)  # Concatenate the sampled token

    print("Raw tokens:", seq_generated)
    return seq_generated




# Example usage:
vocab = Vocabulary(freq_threshold=5)
tokenizer = Tokenizer(vocab, num_classes=6, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
CFG.bos_idx = tokenizer.BOS_code
CFG.pad_idx = tokenizer.PAD_code


encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=64)
decoder = Decoder(vocab_size=305, #complete_vocab_size, #tokenizer.vocab_size,
                  encoder_length=CFG.num_patches, dim=64, num_heads=2, num_layers=2)
model = EncoderDecoder(encoder, decoder)

model.to(CFG.device)

import albumentations as A

def get_transform_inference(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
    ])


import cv2
import albumentations as A
import torch

def preprocess_image_for_inference(image_path, transform):
    """
    Load an image, apply the defined transformations, and prepare it for model inference.
    """
    image = cv2.imread(image_path)  # Load the image with OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Apply transformations
    transformed = transform(image=image)
    transformed_image = transformed["image"]

    # Convert to PyTorch tensor and add batch dimension
    transformed_image = torch.from_numpy(transformed_image).permute(2, 0, 1).unsqueeze(0).float()
    return transformed_image

# Define your size and get the transformation pipeline
size = 224  # Example size, adjust as needed
transform_pipeline = get_transform_inference(size)



# Load the model and weights
model_weights_path = '/mnt/sdb/2024/pix_2_seq_with_captions_march/output_1/best_model_epoch_54.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=CFG.device))

# Process the image(s)
image_path = '/mnt/sdb/2024/pix_2_seq_with_captions_march/images/inclusion_10.jpg'
image_tensor = preprocess_image_for_inference(image_path, transform_pipeline)

# Perform inference
prediction = inference_single_image(model, image_tensor, tokenizer)
print(prediction)

# If you have a second image, repeat the preprocessing and inference steps
