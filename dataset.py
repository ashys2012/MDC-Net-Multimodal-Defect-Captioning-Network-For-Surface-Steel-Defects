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
        #A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        #A.Rotate(limit=30, p=0.5),
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






from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch, max_len, pad_idx):
    image_batch, seq_batch = [], []
    for sample in batch:
        image, seqs = sample[:2]  # Assuming the first two values are always present and seqs is correctly formatted
        image_batch.append(image)

        # Ensure seqs is a flat list; this might need adjustment based on actual structure
        flat_seqs = [item for sublist in seqs for item in sublist] if isinstance(seqs[0], list) else seqs
        seq_batch.append(torch.tensor(flat_seqs, dtype=torch.long))

    # Ensure all sequences are padded to the same length
    seq_batch_padded = pad_sequence(seq_batch, padding_value=pad_idx, batch_first=True)
    return torch.stack(image_batch), seq_batch_padded





    
        # # Pad sequences to the longest sequence in the batch
        # seq_batch_padded = pad_sequence(seq_batch, padding_value=pad_idx, batch_first=True)
        # print(f"seq_batch_padded dimensions in the collate funciton is: {seq_batch_padded.shape}")
        
        
        # # Truncate or further pad the sequences to max_len
        # if max_len and seq_batch_padded.size(1) > max_len:
        #     seq_batch_padded = seq_batch_padded[:, :max_len]
        #     print(f"seq_batch_padded dimensions in the collate funciton is: {seq_batch_padded.shape}")
        # elif max_len and seq_batch_padded.size(1) < max_len:
        #     additional_pad = torch.full((seq_batch_padded.size(0), max_len - seq_batch_padded.size(1)), pad_idx, dtype=torch.long)
        #     print(f"additional_pad dimensions in the collate funciton is: {additional_pad.shape}")
        #     seq_batch_padded = torch.cat([seq_batch_padded, additional_pad], dim=1)

            


        # image_batch = torch.stack(image_batch)
        
        # return image_batch, seq_batch_padded




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