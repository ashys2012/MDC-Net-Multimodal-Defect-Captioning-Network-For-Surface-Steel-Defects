import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import collate_fn, txt_file_to_df
from functools import partial
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from functools import partial
import albumentations as A
import pandas as pd
import cv2
import albumentations as A
import torch
from allied_files import CFG, Tokenizer
from transformers import top_k_top_p_filtering
from model import Encoder, EncoderDecoder, Decoder
import matplotlib.pyplot as plt
import matplotlib.patches as patches  

# Define your category ID to class name mapping here
id2cls = {0: 'crazing', 1: 'inclusion', 2: 'patches', 3: 'pitted_surface', 4: 'rolled-in_scale', 5: 'scratches'}



# txt_file_path = "/home/w19034038/Documents/workspace_trial/one_dd/pix2seq/data/full_neu_bbox_no_captions/annotations.txt"
# image_folder = "/home/w19034038/Documents/workspace_trial/one_dd/pix2seq/data/full_neu_bbox_no_captions/images"

# df = txt_file_to_df(txt_file_path, image_folder)  # Assuming you've already created df using txt_file_to_df
# df['img_path'] = df['img_path'].apply(lambda x: x if x.lower().endswith('.jpg') else f"{x}.jpg")
# train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
# valid_df, test_df = train_test_split(temp_df, test_size=0.20, random_state=42)


# img_paths_from_df = test_df.drop_duplicates(subset=['img_path'])

# class VOCDatasetTest(torch.utils.data.Dataset):
#     def __init__(self, dataframe, size):
#         self.dataframe = dataframe
#         self.transforms = A.Compose([A.Resize(size, size), A.Normalize()])

#     def __getitem__(self, idx):
#         img_path = self.dataframe.iloc[idx]['img_path']
#         print(img_path)

#         img = cv2.imread(img_path)[..., ::-1]

#         if self.transforms is not None:
#             img = self.transforms(image=img)['image']

#         img = torch.FloatTensor(img).permute(2, 0, 1)

#         return img

#     def __len__(self):
#         return len(self.dataframe)

# # Initialize the dataset and DataLoader

# test_dataset = VOCDatasetTest(img_paths_from_df, CFG.img_size)
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=len(img_paths_from_df),  # You can adjust the batch_size as needed
#     shuffle=False,
#     num_workers=0  # You can adjust num_workers based on your system's capabilities
# )


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
    ## sanity check
    invalid_idxs = ((EOS_idxs - 1) % 5 != 0).nonzero().view(-1)
    EOS_idxs[invalid_idxs] = 0
    
    all_bboxes = []
    all_labels = []
    all_confs = []
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0:
            all_bboxes.append(None)
            all_labels.append(None)
            all_confs.append(None)
            continue
        labels, bboxes = tokenizer.decode(batch_preds[i, :EOS_idx+1])
        confs = [round(batch_confs[j][i].item(), 3) for j in range(len(bboxes))]
        
        all_bboxes.append(bboxes)
        all_labels.append(labels)
        all_confs.append(confs)
        
    return all_bboxes, all_labels, all_confs


tokenizer = Tokenizer(num_classes=6, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)


CFG.pad_idx = tokenizer.PAD_code



encoder = Encoder(model_name=CFG.model_name, pretrained=False, out_dim=256)
decoder = Decoder(vocab_size=tokenizer.vocab_size,
                encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
model = EncoderDecoder(encoder, decoder)
model.to(CFG.device)

msg = model.load_state_dict(torch.load('/home/w19034038/Documents/workspace_trial/one_dd/pix2seq/checkpoint/last_model.pth', map_location=CFG.device))
print(msg)
model.eval()





img_paths = """random.jpg crazing_299.jpg crazing_300.jpg inclusion_299.jpg inclusion_300.jpg patches_299.jpg patches_300.jpg pitted_surface_299.jpg pitted_surface_300.jpg rolled-in_scale_299.jpg rolled-in_scale_300.jpg scratches_299.jpg scratches_300.jpg"""
img_paths = ["/home/w19034038/Documents/workspace_trial/one_dd/pix2seq/data/full_neu_bbox_no_captions/test_imagees/" + path for path in img_paths.split(" ")]



class VOCDatasetTest(torch.utils.data.Dataset):
    def __init__(self, img_paths, size):
        self.img_paths = img_paths
        self.transforms = A.Compose([A.Resize(size, size), A.Normalize()])

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(img_path)[..., ::-1]

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
all_confs = []

with torch.no_grad():
    for i, x in enumerate(tqdm(test_loader)):
        print(f"Batch {i+1}:")
        print(f"Data Shape: {x.shape}")  # If x is a tensor
        # If x is a list or tuple of tensors (multi-input or include labels)
        # print([xi.shape for xi in x])

        batch_preds, batch_confs = generate(
            model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
        bboxes, labels, confs = postprocess(
            batch_preds, batch_confs, tokenizer)
        
        # Print or inspect bboxes, labels, and confs for the current batch
        print("Bboxes:", bboxes)
        print("Labels:", labels)
        print("Confs:", confs)
        
        all_bboxes.extend(bboxes)
        all_labels.extend(labels)
        all_confs.extend(confs)


for i, (bboxes, labels, confs) in enumerate(zip(all_bboxes, all_labels, all_confs)):
    img_path = img_paths[i]
    img = cv2.imread(img_path)[..., ::-1]
    img = cv2.resize(img, (CFG.img_size, CFG.img_size))
    img = visualize(img, bboxes, labels, id2cls, show=False)

    cv2.imwrite("/home/w19034038/Documents/workspace_trial/one_dd/pix2seq/output_folder/" + img_path.split("/")[-1], img[..., ::-1])