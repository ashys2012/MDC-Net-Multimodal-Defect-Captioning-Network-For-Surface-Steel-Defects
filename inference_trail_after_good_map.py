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

# Define your category ID to class name mapping here
id2cls = {258: 'crazing', 259: 'inclusion', 260: 'patches', 261: 'pitted_surface', 262: 'rolled-in_scale', 263: 'scratches'}


def generate(model, x, tokenizer, max_len=50, top_k=0, top_p=1):
    x = x.to(CFG.device)
    batch_preds = torch.ones(x.size(0), 1).fill_(tokenizer.BOS_code).long().to(CFG.device)
    confs = []
    
    if top_k != 0 or top_p != 1:
        sample = lambda preds: torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)
        
    with torch.no_grad():
        for i in range(max_len):
            preds = model.predict(x, batch_preds)
            ## If top_k and top_p are set to default, the following line does nothing!
            preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)
            if i % 4 == 0:
                confs_ = torch.softmax(preds, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                confs.append(confs_)
            preds = sample(preds)
            batch_preds = torch.cat([batch_preds, preds], dim=1)
    
    return batch_preds.cpu(), confs


def postprocess(batch_preds, batch_confs, tokenizer):
    EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
    invalid_idxs = ((EOS_idxs - 1) % 5 != 0).nonzero().view(-1)
    EOS_idxs[invalid_idxs] = 0
    
    all_bboxes = []
    all_labels = []
    all_captions = []  # Added this line to store captions
    all_confs = []
    
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0:
            all_bboxes.append(None)
            all_labels.append(None)
            all_captions.append(None)  # Added this line to store None for captions
            all_confs.append(None)
            continue
        
        labels, bboxes, captions = tokenizer.decode(batch_preds[i, :EOS_idx+1])
        confs = [round(batch_confs[j][i].item(), 3) for j in range(len(bboxes))]
        
        all_bboxes.append(bboxes)
        all_labels.append(labels)
        all_captions.append(captions)  # Added this line to store captions
        all_confs.append(confs)
        
    return all_bboxes, all_labels, all_captions, all_confs


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


encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=64)
decoder = Decoder(vocab_size=305, #complete_vocab_size, #tokenizer.vocab_size,
                  encoder_length=CFG.num_patches, dim=64, num_heads=2, num_layers=2)
model = EncoderDecoder(encoder, decoder)

model.to(CFG.device)


msg = model.load_state_dict(torch.load('/mnt/sdb/2024/pix_2_seq_with_captions_march/output_1/best_model_epoch_156.pth', map_location=CFG.device))
print(msg)
model.eval()





img_paths = """inclusion_2.jpg"""
img_paths = ["/mnt/sdb/2024/pix_2_seq_with_captions_march/images/" + path for path in img_paths.split(" ")]



class VOCDatasetTest(torch.utils.data.Dataset):
    def __init__(self, img_paths, size):
        self.img_paths = img_paths
        self.transforms = A.Compose([A.Resize(size, size), A.Normalize()])

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        print("Image Path:", img_path)

        img = cv2.imread(img_path)[..., ::-1]
        if img is None:
            print(f"Image not found at {img_path}")
            return

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        img = torch.FloatTensor(img).permute(2, 0, 1)

        return img

    def __len__(self):
        return len(self.img_paths)
    

test_dataset = VOCDatasetTest(img_paths, size=CFG.img_size)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=2, shuffle=False, num_workers=0)

for batch in test_loader:
    print("Batch shape:", batch.shape) 

GT_COLOR = (0, 255, 0) # Green
PRED_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color, thickness=1):
    """Visualizes a single bounding box on the image"""
    bbox = [int(item) for item in bbox]
    x_min, y_min, x_max, y_max = bbox
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min), (x_min + text_width, y_min + int(text_height * 1.3)), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min+ int(text_height * 1.3)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name, color=PRED_COLOR, show=True):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, color)
    if show:
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    return img


all_bboxes = []
all_labels = []
all_captions = []
all_confs = []

with torch.no_grad():
    for i, x in enumerate(tqdm(test_loader)):
        print(f"Batch {i+1}:")
        print(f"Data Shape: {x.shape}")

        batch_preds, batch_confs = generate(
            model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
        bboxes, labels, captions, confs = postprocess(  # Modified this line to include captions
            batch_preds, batch_confs, tokenizer)
        
        print("Bboxes:", bboxes)
        print("Labels:", labels)
        print("Captions:", captions)  # Added this line to print captions
        print("Confs:", confs)
        
        all_bboxes.extend(bboxes)
        all_labels.extend(labels)
        all_captions.extend(captions)  # Added this line to extend all_captions
        all_confs.extend(confs)


for i, (bboxes, labels, confs) in enumerate(zip(all_bboxes, all_labels, all_confs)):
    img_path = img_paths[i]
    img = cv2.imread(img_path)[..., ::-1]
    img = cv2.resize(img, (CFG.img_size, CFG.img_size))
    img = visualize(img, bboxes, labels, id2cls, show=False)

    cv2.imwrite("/mnt/sdb/2024/pix_2_seq_with_captions_march/output_images_after_test/" + img_path.split("/")[-1], img[..., ::-1])