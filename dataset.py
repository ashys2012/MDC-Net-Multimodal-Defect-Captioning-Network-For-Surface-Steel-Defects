'''
THis code takes in single bbox dataset and converts it into dataframe that is specified in the 
https://colab.research.google.com/drive/1UeYIZ6_GHNwCHSi8nNV5dVc3oUrM5-BA?usp=sharing#scrollTo=knCEq7-qLAZZ

THis code works and is used for Pix2Seq


'''
import albumentations as A
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import os
import torch
from torch.nn.utils.rnn import pad_sequence


# # To display all columns
# pd.set_option('display.max_columns', None)

# # To display all rows
# pd.set_option('display.max_rows', None)


# # To display the entire contents of each cell (useful for large strings)
# pd.set_option('display.max_colwidth', None)


import os
import pandas as pd

def txt_file_to_df(txt_file_path, image_folder):
    ids = []
    captions = []
    labels = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    img_paths = []
    
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()[1:] 
        for line in lines:
            #print(f"Reading line: {line.strip()}")  # Debug print
            parts = line.strip().split(',')
            #print(f"Parts: {parts}")  # Debug print
            if len(parts) < 7:  # Check for all required parts except caption
                print(f"Skipping malformed line: {line}")
                continue

            image_name = parts[0]
            image_id = parts[1]
            label = int(parts[2])
            coords = list(map(int, parts[3:7]))
            
            # Handle optional caption
            caption = parts[7] if len(parts) > 7 else "No caption"
            
            #print(f"Extracted caption: {caption}") 
                
            if len(coords) < 4:
                print(f"Skipping line with insufficient coordinates: {line}")
                continue
            
            ids.append(image_id)
            captions.append(caption)
            labels.append(label)
            xmin.append(coords[0])
            ymin.append(coords[1])
            xmax.append(coords[2])
            ymax.append(coords[3])
            img_paths.append(os.path.join(image_folder, image_name))
            
    df = pd.DataFrame({
        'ids': ids,
        'caption': captions,
        'label': labels,
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
        'img_path': img_paths
    })
    
    print("this is the main df ----------", df.tail(5))
    return df




def get_transform_train(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MotionBlur(blur_limit=3, p=0.5),
        #A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        #A.GridDistortion(p=0.5),
        #A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), p=0.5),
        A.Resize(size, size),
        A.Normalize(),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})




def get_transform_valid(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})






def collate_fn(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length
    """
    # print("Number of samples in batch:", len(batch))
    # for i, sample in enumerate(batch):
    #     print(f"Number of elements in sample {i}: {len(sample)}")
        #print(f"Types of elements in sample {i}: {[type(x) for x in sample]}")
    # rest of your code

    image_batch, seq_batch = [], []
    for sample in batch:
        #print("Received batch lengths:", [len(sample) for sample in batch])
        #print("The batch is ",batch)
        if len(sample) == 2:
            image, seqs = sample
            #caption = None  # or some default value
        elif len(sample) == 4:
            image, labels, bboxes, caption = sample
            seqs = ...  # Generate seqs based on labels and bboxes
        else:
            raise ValueError("Unexpected number of values in sample")
        #print("Received batch:", batch)
        image_batch.append(image)
        seq_batch.append(seqs)
        #caption_batch.append(caption)

    seq_batch = pad_sequence(
        seq_batch, padding_value=pad_idx, batch_first=True)
    if max_len:
        pad = torch.ones(seq_batch.size(0), max_len -
                         seq_batch.size(1)).fill_(pad_idx).long()
        seq_batch = torch.cat([seq_batch, pad], dim=1)
    image_batch = torch.stack(image_batch)
    #print("The seq_batch is inside the collate fucntion is", seq_batch)
    #print("The seq_batch shape is inside the collate fucntion is", seq_batch.shape)

    return image_batch, seq_batch#, caption_batch



# def check_bbox_coordinates(txt_file_path):
#     incorrect_lines = []
    
#     with open(txt_file_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             parts = line.strip().split(',')
#             xmin, ymin, xmax, ymax = map(int, parts[-4:])
            
#             if xmax <= xmin or ymax <= ymin:
#                 incorrect_lines.append(line.strip())
                
#     if incorrect_lines:
#         print("Lines with incorrect bounding box coordinates:")
#         for line in incorrect_lines:
#             print(line)
#     else:
#         print("All bounding box coordinates are correct.")



# if __name__ == '__main__':
#     txt_file_path = '/home/w19034038/Documents/workspace_trial/one_dd/pix2seq/data/full_neu_bbox_no_captions/annotations.txt'  # Replace with the path to your text file
#     check_bbox_coordinates(txt_file_path)
    
    # Your existing code for data loading, training, etc.



# if __name__ == "__main__":
#     print("Names inside the module:", dir())
