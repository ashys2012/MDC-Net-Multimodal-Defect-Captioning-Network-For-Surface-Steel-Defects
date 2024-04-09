import numpy as np
from collections import Counter
import spacy
import torch
from sklearn.model_selection import train_test_split
from dataset import get_transform_train, get_transform_valid, collate_fn
import os
from functools import partial
import cv2
from allied_files import CFG
spacy_eng = spacy.load('en_core_web_sm') 
from torch.nn.utils.rnn import pad_sequence


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {302: "<PAD>", 300: "<SOS>", 301: "<EOS>", 299: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

        # Predefined indices for specific words
        self.predefined_indices = {
            'oil_spot': 262,
            'inclusion': 264,
            'crescent_gap': 260,
            'water_spot': 261,
            'punching_hole': 258,
            'welding_line': 259,
            'silk_spot': 263,
            'rolled_pit': 265,
            'crease': 266,
            'waist_folding': 267
        }

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        compound_words = ['inclusion','rolled-oil_spot', 'crescent_gap', 'water_spot', 'water_spot', 'punching_hole', 'welding_line', 'silk_spot', 'rolled_pit', 'crease', 'waist_folding']

        # Replace compound words with placeholders
        placeholders = {}
        for compound in compound_words:
            placeholder = compound.replace('-', '_').replace('_', '') # Create a unique placeholder
            placeholders[placeholder] = compound
            text = text.replace(compound, placeholder)
        #rolled_in-scale will look like rolledinscale after the above code
        
        # Use spacy tokenizer
        tokens = [token.text.lower() for token in spacy_eng.tokenizer(text)]

        # Replace placeholders back with original compound words
        tokens = [placeholders.get(token, token) for token in tokens]
        return tokens
        #return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 270  # Starting index for non-special tokens

        # First, add predefined words with their specific indices
        for word, predefined_idx in self.predefined_indices.items():
            self.stoi[word] = predefined_idx
            self.itos[predefined_idx] = word

        # Update starting index if predefined indices are in the way
        while idx in self.itos:
            idx += 1

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

        for word, count in frequencies.items():
            if count >= self.freq_threshold and word not in self.predefined_indices:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        text = str(text)  # Convert numpy.str_ to str if needed
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
    
    def decode(self, numericalized_sentence):
        # Map each numerical ID back to its corresponding word
        decoded_sentence = [self.itos.get(id, "<UNK>") for id in numericalized_sentence]
        # Join the words back into a single string
        return ' '.join(decoded_sentence)



vocab = Vocabulary(freq_threshold=5)


import os
import cv2
import torch
from torch.utils.data import Dataset




import torch
import pandas as pd
import os
import cv2

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None, tokenizer=None):
        self.entries = self._flatten_dataframe(df)
        self.transforms = transforms
        self.tokenizer = tokenizer

    def _flatten_dataframe(self, df):
        entries = []
        for _, row in df.iterrows():
            # Directly create a tuple from the bounding box coordinates
            bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            entries.append({
                'img_path': row['img_path'],
                'bbox': bbox,
                'caption': row['caption'],
                'label': row['label']  # If you need the label
            })
        return entries


    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = entry['img_path']
        if not os.path.exists(img_path):
            print(f"Warning: The file '{img_path}' does not exist. Skipping.")
            return None

        img = cv2.imread(img_path)[..., ::-1]  # Convert BGR to RGB
        bbox = [entry['bbox']]  # Wrap bbox in a list to match expected format
        caption = [entry['caption']]  # Similarly, wrap caption in a list
        label = [entry['label']]  # Wrap label in a list

        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=bbox, labels=label)
            img = transformed['image']
            bbox = transformed['bboxes']

        img = torch.FloatTensor(img).permute(2, 0, 1)  # Convert to torch tensor and permute to CxHxW

        # Now call the tokenizer with all required arguments
        if self.tokenizer is not None:
            sequences = self.tokenizer(labels=label, bboxes=bbox, captions=caption)
        else:
            sequences = []

        # Note: Adjust the return statement based on how you intend to use `sequences` and what it contains
        return img, sequences

    def __len__(self):
        return len(self.entries)





def get_loaders(df, tokenizer, img_size, batch_size, max_len, pad_idx, num_workers=2, valid_size=0.15, test_size=0.05):
    # Adjust valid_size to account for the size after removing the test set
    # This calculation ensures that the validation set is 15% of the original dataset,
    # and the test set is 5%, leaving the training set with 80%.
    valid_size_adj = valid_size / (1 - test_size)

    # First split: separate out the test dataset
    train_valid_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Second split: split the remaining data into train and validation datasets
    train_df, valid_df = train_test_split(train_valid_df, test_size=valid_size_adj, random_state=42)

    print("Training dataset size: ", len(train_df))
    print("Validation dataset size: ", len(valid_df))
    print("Test dataset size: ", len(test_df))
    
    # Define datasets for each split
    train_ds = VOCDataset(train_df, transforms=get_transform_train(img_size), tokenizer=tokenizer)
    valid_ds = VOCDataset(valid_df, transforms=get_transform_valid(img_size), tokenizer=tokenizer)
    test_ds = VOCDataset(test_df, transforms=get_transform_valid(img_size), tokenizer=tokenizer)

    print("train_ds", train_ds[0])

    # DataLoader for the training set
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    # DataLoader for the validation set
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    # DataLoader for the test set
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=40, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )
    print("length of train_loader is", len(train_loader))

    return train_loader, valid_loader, test_loader




class Tokenizer:
    def __init__(self, vocab, num_classes: int, num_bins: int, width: int, height: int, max_len=200, caption_length=40):
        self.vocab = vocab
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.width = width
        self.height = height
        self.max_len = max_len
        self.caption_length = caption_length
        
        # Existing codes
        self.BOS_code = 300
        self.EOS_code = self.BOS_code + 1              #301
        self.PAD_code = self.EOS_code + 1              #302
        self.CAPTION_START = self.PAD_code + 1         #303
        self.CAPTION_END = self.CAPTION_START + 1      #304

        # New tokens
        #self.BOB_code = self.CAPTION_END + 1  # Beginning of Bounding Box
        #self.EOB_code = self.BOB_code + 1       # End of Bounding Box


        # Update vocab size
        self.vocab_size = self.CAPTION_END + 1

        #self.vocab_size = num_classes + num_bins + 3
    
    
    def quantize(self, x: np.array):
        """
        x is a real number in [0, 1]
        """
        return (x * (self.num_bins - 1)).astype('int')
    
    def dequantize(self, x: np.array):
        """
        x is an integer between [0, num_bins-1]
        """
        return x.astype('float32') / (self.num_bins - 1)

    def process_single_pair(self, label, bbox, caption):
        bbox = np.array(bbox, dtype=float)

        # Normalize and quantize bbox coordinates
        bbox[0] = bbox[0] / self.width
        bbox[2] = bbox[2] / self.width
        bbox[1] = bbox[1] / self.height
        bbox[3] = bbox[3] / self.height

        tokenized = [self.BOS_code]

        # Tokenize and append the corresponding caption with start and end tokens
        caption_tokens = [self.CAPTION_START]  # Start of caption
        caption_tokens.extend(self.vocab.numericalize(caption))
        caption_tokens.append(self.CAPTION_END)  # End of caption
        tokenized.extend(caption_tokens)

        # Tokenize and append the label
        tokenized.append(label)

        # Extend the tokenized list with the quantized bbox coordinates
        quantized_bbox = self.quantize(np.array(bbox, dtype=float))
        tokenized.extend(map(int, quantized_bbox))

        tokenized.append(self.EOS_code)

        return tokenized[:self.max_len]

    def __call__(self, labels, bboxes, captions):
        assert len(captions) == len(bboxes) == len(labels), "Each bbox must have a corresponding label and caption"
        
        sequences = []
        for label, bbox, caption in zip(labels, bboxes, captions):
            sequence = self.process_single_pair(label, bbox, caption)
            sequences.append(sequence)
        
        return sequences


    
    def get_spacy_vocab_size(self):
        return len(self.vocab)

    def get_complete_vocab_size(self):
        return self.vocab_size

    
    def print_vocab(self):
        print("STOI:", self.vocab.stoi)
        print("ITOS:", self.vocab.itos)
    
        

    def decode(self, tokens):
        # Convert list to PyTorch tensor if necessary
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=CFG.device)

        tokens = tokens.clone().detach()
        
        if tokens.numel() == 0:
            print("Empty tokens tensor")
            return [], [], ""
        
        # If tokens tensor is scalar, add a dimension
        if tokens.dim() == 0:               
            tokens = tokens.unsqueeze(0)
        
        # Remove only PAD tokens initially
        tokens = tokens[tokens != self.PAD_code]

        # Find the EOS token's index
        eos_idx = (tokens == self.EOS_code).nonzero(as_tuple=True)[0]   # this gets the first occurance of the EOS token



        # If EOS token is found, keep only the tokens before EOS
        if eos_idx.nelement() > 0:
            eos_idx = eos_idx[0].item()
            tokens = tokens[:eos_idx]
            #print("After truncating at EOS token:", tokens.tolist())

        #print("The tokens in the tokenizer are", tokens)

        labels, bboxes = [], []
        captions_text = ""
        
        # Find SOC and EOC indices
        soc_idxs = (tokens == self.CAPTION_START).nonzero(as_tuple=True)[0]
        soc_idx = soc_idxs[0].item() if soc_idxs.nelement() > 0 else None
        eoc_idxs = (tokens == self.CAPTION_END).nonzero(as_tuple=True)[0]
        eoc_idx = eoc_idxs[0].item() if eoc_idxs.nelement() > 0 else None

        if soc_idx is not None and eoc_idx is not None:
            # Extract caption
            caption_tokens = tokens[soc_idx+1:eoc_idx]
            captions_text = self.tokens_to_text(caption_tokens.tolist())

            # Process bounding boxes and labels after EOC, up to EOS
            bbox_label_tokens = tokens[eoc_idx+1:]       #if eos is there then the code will be bbox_label_tokens = tokens[eoc_idx+1:eos_idx]

            for i in range(0, len(bbox_label_tokens), 5):
                if i+4 < len(bbox_label_tokens):
                    label_tensor = bbox_label_tokens[i:i+1]
                    bbox = bbox_label_tokens[i+1:i+5]
                    
                    # Check for valid label and bbox values
                    if label_tensor.numel() == 1 and label_tensor.item() in range(258, 268) and all(0 <= item <= 224 for item in bbox):
                        label = label_tensor.item()
                        labels.append(label)
                        bboxes.append([int(item) for item in bbox])
                    else:
                        pass
                        #print(f"Invalid label or bbox values: {label_tensor} {bbox}")

        # Adjust BBoxes to original dimensions
        bboxes = np.array(bboxes, dtype=float)
        if bboxes.size > 0:
            if bboxes.ndim == 1:
                bboxes[0] = self.dequantize(bboxes[0]) * self.width
                bboxes[2] = self.dequantize(bboxes[2]) * self.width
                bboxes[1] = self.dequantize(bboxes[1]) * self.height
                bboxes[3] = self.dequantize(bboxes[3]) * self.height
            elif bboxes.ndim == 2:
                bboxes[:, [0, 2]] = self.dequantize(bboxes[:, [0, 2]]) * self.width
                bboxes[:, [1, 3]] = self.dequantize(bboxes[:, [1, 3]]) * self.height

        return labels, bboxes.tolist(), captions_text
    


    def decode_captions(self, tokens):
        # Convert list to PyTorch tensor if necessary
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=CFG.device)

        tokens = tokens.clone().detach()
        
        if tokens.numel() == 0:
            print("Empty tokens tensor")
            return [], [], ""
        
        # If tokens tensor is scalar, add a dimension
        if tokens.dim() == 0:               
            tokens = tokens.unsqueeze(0)
        
        # Remove only PAD tokens initially
        tokens = tokens[tokens != self.PAD_code]

        # Find the End of caption token's index
        eoc_idx = (tokens == self.CAPTION_END).nonzero(as_tuple=True)[0]   # this gets the first occurance of the EOS token
        # If EOS token is found, keep only the tokens before EOS
        if eoc_idx.nelement() > 0:
            eoc_idx = eoc_idx[0].item()
            tokens = tokens[:eoc_idx]

        # Find the Start of caption token's index
        boc_idx = (tokens == self.CAPTION_START).nonzero(as_tuple=True)[0]   # this gets the first occurance of the EOS token
        # If EOS token is found, keep only the tokens before EOS
        if boc_idx.nelement() > 0:
            boc_idx = boc_idx[0].item() + 1
            tokens = tokens[boc_idx:]
        return tokens





    # def decode_bboxes(self, pred_seq, caption_end_token=304, label_start=258, label_end=263, eos_token=301):

    #     if isinstance(pred_seq, list):
    #         pred_seq = torch.tensor(pred_seq, device=CFG.device)

    #     pred_seq = pred_seq.clone().detach()

        
    #     if pred_seq.numel() == 0:
    #         return [], [], ""
        
    #     # If tokens tensor is scalar, add a dimension
    #     if pred_seq.dim() == 0:               
    #         pred_seq = pred_seq.unsqueeze(0)
        
    #     # Remove only PAD tokens initially
    #     #pred_seq = pred_seq[pred_seq != self.PAD_code]
    #     """
    #     Decode bounding boxes from the predicted sequence, ensuring they follow a label token
    #     and are within the valid range and structure: caption end, label, bbox coordinates, ..., EOS.
    #     Outputs a padded 3D tensor for compatibility with batches having variable numbers of bounding boxes.
        
    #     :param pred_seq: Tensor of predicted sequences (batch_size, sequence_length).
    #     :param caption_end_token: The token indicating the end of captions (EOC).
    #     :param label_start: The starting index for label tokens.
    #     :param label_end: The ending index for label tokens.
    #     :param eos_token: The EOS token indicating the end of the sequence.
    #     :return: A 3D tensor containing decoded bboxes and labels, organized per image, with padding.
    #     """
    #     all_decoded_bboxes = []


    #     for seq in pred_seq:
    #         decoded_bboxes = []
    #         # Find the end of the caption
    #         eoc_idx = (seq == caption_end_token).nonzero(as_tuple=True)[0]
    #         if len(eoc_idx) > 0:
    #             start_idx = eoc_idx[0].item() + 1  # Start after the caption end token
    #         else:
    #             start_idx = 0  # Default to start if EOC token is not found

    #         i = start_idx
    #         while i < len(seq) - 4:  # Ensure room for label + bbox coordinates
    #             token = seq[i].item()
    #             # Check if the token is a label token
    #             if label_start <= token <= label_end:
    #                 # Extract potential bbox coordinates following the label
    #                 bbox = seq[i + 1:i + 5]
    #                 # Validate bbox: ensure coordinates are within expected range and form a valid bbox
    #                 if torch.all(bbox >= 0) and torch.all(bbox <= 224) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
    #                     decoded_bboxes.append(bbox)
    #                 i += 5  # Move past this bbox sequence
    #             elif token == eos_token:
    #                 break  # End of sequence
    #             else:
    #                 i += 1  # Continue to next token if not a label or EOS
    #         #print("The decoded bboxes are", decoded_bboxes)        #-----here the bbox is in consistent format
    #         # Convert list of tensors to a tensor for current sequence
    #         if decoded_bboxes:
    #             decoded_bboxes = torch.stack(decoded_bboxes)
    #         else:
    #             # Use a dummy tensor with shape [1, 4] filled with zeros if no bboxes are found
    #             decoded_bboxes = torch.zeros(1, 4)
    #         #print("The decoded bboxes are", decoded_bboxes)       #----> here the bbox is in consistent format    
    #         all_decoded_bboxes.append(decoded_bboxes)
    #         print("The decoded bboxes are", all_decoded_bboxes)   #---> gives wrong grousn truth bbox
    #         for i, bboxes_tensor in enumerate(all_decoded_bboxes):
    #             if bboxes_tensor.size(0) > 1:  # Check if there are decoded bboxes
    #                 bboxes_np = bboxes_tensor.cpu().numpy() 
    #                 bboxes_dequantized_np = self.dequantize(bboxes_np)
    #                 all_decoded_bboxes[i] = torch.tensor(bboxes_dequantized_np, device=bboxes_tensor.device)
    #             else:
    #                 # Handle the case where there are no bboxes or only dummy bboxes
    #                 all_decoded_bboxes[i] = torch.zeros_like(bboxes_tensor).float()

            

    #     # Pad sequences to have the same length
    #     padded_bboxes = pad_sequence(all_decoded_bboxes, batch_first=True, padding_value=0)
    #     #print("The padded bboxes are", padded_bboxes)

    #     return padded_bboxes

  

    

    def decode_labels(self, tokens):
        PAD_TOKEN = CFG.pad_idx
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=CFG.device)

        tokens = tokens.clone().detach()
        
        if tokens.numel() == 0:
            return []
        
        # If tokens tensor is scalar, add a dimension
        if tokens.dim() == 0:               
            tokens = tokens.unsqueeze(0)        
        
        LABEL_START, LABEL_END = 258, 267
        label_mask = (tokens >= LABEL_START) & (tokens <= LABEL_END)

        # Applying mask to isolate label tokens, maintaining structure
        label_tokens = tokens * label_mask

        # Iterate over batch to extract the first label token per item
        first_labels = []
        for item_labels in label_tokens:
            item_labels = item_labels[item_labels.nonzero(as_tuple=True)]  # Filter non-zero labels
            first_label = item_labels[0] if len(item_labels) > 0 else PAD_TOKEN  # Take first or None if no labels
            first_labels.append(first_label)
        return torch.tensor(first_labels, device=tokens.device)
    
    def adjust_bboxes_dimensions(self, bboxes_tensor):
        """
        Adjusts dequantized bounding boxes to their original dimensions.
        """
        bboxes_dequantized_np = self.dequantize(bboxes_tensor.cpu().numpy())
        bboxes_dequantized_np[:, [0, 2]] = bboxes_dequantized_np[:, [0, 2]] * self.width
        bboxes_dequantized_np[:, [1, 3]] = bboxes_dequantized_np[:, [1, 3]] * self.height
        return torch.tensor(bboxes_dequantized_np, device=bboxes_tensor.device).float()

    def decode_bboxes(self, pred_seq, caption_end_token=304, label_start=258, label_end=267, eos_token=301):
        if isinstance(pred_seq, list):
            pred_seq = torch.tensor(pred_seq, device=CFG.device)

        pred_seq = pred_seq.clone().detach()

        if pred_seq.numel() == 0:
            return [], [], ""

        if pred_seq.dim() == 0:
            pred_seq = pred_seq.unsqueeze(0)

        all_decoded_bboxes = []

        for seq in pred_seq:
            decoded_bboxes = []
            eoc_idx = (seq == caption_end_token).nonzero(as_tuple=True)[0]
            start_idx = eoc_idx[0].item() + 1 if len(eoc_idx) > 0 else 0

            i = start_idx
            while i < len(seq) - 4:
                token = seq[i].item()
                if label_start <= token <= label_end:
                    bbox = seq[i + 1:i + 5]
                    if torch.all(bbox >= 0) and torch.all(bbox <= 224) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        decoded_bboxes.append(bbox)
                    i += 5
                elif token == eos_token:
                    break
                else:
                    i += 1

            if decoded_bboxes:
                decoded_bboxes = torch.stack(decoded_bboxes)
                # Adjust bounding box dimensions directly after decoding
                adjusted_bboxes = self.adjust_bboxes_dimensions(decoded_bboxes)
                all_decoded_bboxes.append(adjusted_bboxes)
            else:
                all_decoded_bboxes.append(torch.zeros(1, 4, device=CFG.device))

        padded_bboxes = pad_sequence(all_decoded_bboxes, batch_first=True, padding_value=0)
        #print("The padded bboxes are", padded_bboxes)
        return padded_bboxes


    def decode_bboxes_and_labels_with_scores(self, pred_seq, pred_scores, caption_end_token=304, label_start=258, label_end=267, eos_token=301):
        # Ensure pred_seq is a tensor
        if isinstance(pred_seq, list):
            pred_seq = torch.tensor(pred_seq, device=CFG.device)
        
        # Same for pred_scores
        if isinstance(pred_scores, list):
            pred_scores = torch.tensor(pred_scores, device=CFG.device)
        
        pred_seq = pred_seq.clone().detach()
        pred_scores = pred_scores.clone().detach()

        if pred_seq.numel() == 0:
            return [], [], [], ""

        if pred_seq.dim() == 0:
            pred_seq = pred_seq.unsqueeze(0)
            pred_scores = pred_scores.unsqueeze(0)

        all_decoded_bboxes = []
        all_labels = []
        all_scores = []

        for seq, scores in zip(pred_seq, pred_scores):
            decoded_bboxes = []
            labels = []
            bbox_scores = []
            
            eoc_idx = (seq == caption_end_token).nonzero(as_tuple=True)[0]
            start_idx = eoc_idx[0].item() + 1 if len(eoc_idx) > 0 else 0

            i = start_idx
            while i < len(seq) - 4:
                token = seq[i].item()
                if label_start <= token <= label_end:
                    bbox = seq[i + 1:i + 5]
                    if torch.all(bbox >= 0) and torch.all(bbox <= 224) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        decoded_bboxes.append(bbox)
                        labels.append(token)  # Capture the label
                        
                        # Average the scores for the bbox coordinates
                        bbox_score = scores[i + 1:i + 5].mean().item()
                        bbox_scores.append(bbox_score)
                        
                    i += 5
                elif token == eos_token:
                    break
                else:
                    i += 1

            if decoded_bboxes:
                decoded_bboxes = torch.stack(decoded_bboxes)
                labels = torch.tensor(labels, device=CFG.device)
                all_decoded_bboxes.append(decoded_bboxes)
                all_labels.append(labels)
                all_scores.append(torch.tensor(bbox_scores, device=CFG.device))
            else:
                all_decoded_bboxes.append(torch.zeros(1, 4, device=CFG.device))
                all_labels.append(torch.tensor([], device=CFG.device, dtype=torch.int64))
                all_scores.append(torch.tensor([], device=CFG.device))

        padded_bboxes = pad_sequence(all_decoded_bboxes, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(all_labels, batch_first=True, padding_value=-1)
        padded_scores = pad_sequence(all_scores, batch_first=True, padding_value=-1)

        return padded_bboxes, padded_labels, padded_scores


    def decode_bboxes_and_labels(self, pred_seq, caption_end_token=304, label_start=258, label_end=267, eos_token=301):
        if isinstance(pred_seq, list):
            pred_seq = torch.tensor(pred_seq, device=CFG.device)

        pred_seq = pred_seq.clone().detach()

        if pred_seq.numel() == 0:
            return [], [], ""

        if pred_seq.dim() == 0:
            pred_seq = pred_seq.unsqueeze(0)

        all_decoded_bboxes = []
        all_labels = []

        for seq in pred_seq:
            decoded_bboxes = []
            labels = []
            eoc_idx = (seq == caption_end_token).nonzero(as_tuple=True)[0]
            start_idx = eoc_idx[0].item() + 1 if len(eoc_idx) > 0 else 0

            i = start_idx
            while i < len(seq) - 4:
                token = seq[i].item()
                if label_start <= token <= label_end:
                    bbox = seq[i + 1:i + 5]
                    if torch.all(bbox >= 0) and torch.all(bbox <= 224) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        decoded_bboxes.append(bbox)
                        labels.append(token)  # Capture the label
                    i += 5
                elif token == eos_token:
                    break
                else:
                    i += 1

            if decoded_bboxes:
                decoded_bboxes = torch.stack(decoded_bboxes)
                labels = torch.tensor(labels, device=CFG.device)
                all_decoded_bboxes.append(decoded_bboxes)
                all_labels.append(labels)
            else:
                all_decoded_bboxes.append(torch.zeros(1, 4, device=CFG.device))
                all_labels.append(torch.tensor([], device=CFG.device, dtype=torch.int64))

        padded_bboxes = pad_sequence(all_decoded_bboxes, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(all_labels, batch_first=True, padding_value=-1)  # Use -1 or any appropriate value for padding

        return padded_bboxes, padded_labels


    #you can get alternate approach to extract the predicted labels in https://www.notion.so/Alternate-Label_loss-488ab964219a4f00a9fa22e066bc3886
    #this alternate aproach does not have pad_idx and uses 0 score probability to get the label
    def extract_predicted_labels_with_logits(self, logits):
        PAD_TOKEN = CFG.pad_idx
        LABEL_START, LABEL_END = 258, 267
        
        batch_size, seq_len, num_classes = logits.shape
        # Prepare tensors for extracted logits and labels
        extracted_logits = torch.empty(batch_size, num_classes, device=logits.device)
        extracted_label_indices = torch.empty(batch_size, dtype=torch.long, device=logits.device)
        
        # Iterate over each sequence in the batch
        for i in range(batch_size):
            # Directly work with logits to find the first label position
            # This step might involve using domain-specific knowledge about how labels are encoded in logits
            
            # Example approach (needs adaptation to your specific case):
            # Assuming label token indices are directly reflected in the logits in some manner
            # If logits do not directly encode label positions, adjust this logic accordingly
            label_mask = torch.zeros(seq_len, dtype=torch.bool, device=logits.device)
            for j in range(LABEL_START, LABEL_END + 1):
                label_mask |= logits[i, :, j].nonzero(as_tuple=True)[0].bool()
            
            label_idx = label_mask.nonzero(as_tuple=True)[0]
            
            if len(label_idx) > 0:
                # If a label position is identified, select its logits
                first_label_pos = label_idx[0]
                extracted_logits[i] = logits[i, first_label_pos]
                # This assumes a separate process to accurately obtain label indices if necessary
            else:
                # Handle sequences without identifiable labels
                extracted_logits[i].fill_(PAD_TOKEN)
                # Indicate no label found; adjust according to how you wish to handle such cases
        
        # This function now focuses on extracting logits and does not directly provide label indices
        # Adjust usage accordingly in loss calculation
        return extracted_logits


        
    def tokens_to_text(self, captions):
        # If captions is empty, return an empty list
        if not captions:
            return []

        # If captions is a list of integers, convert it to a list of lists of integers
        if isinstance(captions[0], int):
            captions = [[caption] for caption in captions]

        # Convert list of token lists back to list of caption texts
        return [" ".join([self.vocab.itos.get(token, '<UNK>') for token in caption]) for caption in captions]
    
    #below is an alternate fucntion created on 22nd March to look at the tokens to text
    
    # Define a function to convert tokens to text
    def tokens_to_text_new(self, tokens_list, itos):
        if not tokens_list:
            return []
        
        # If tokens_list is a list of integers, wrap it in another list
        if isinstance(tokens_list[0], int):
            tokens_list = [tokens_list]

        return [' '.join([itos[token] for token in tokens if itos[token] not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']]) for tokens in tokens_list]






    

def top_k_sampling(logits, k):
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = -float('Inf')
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)

def extract_tokens(pred_probs):
    """
    Extract the most probable tokens from the predicted probabilities.

    :param pred_probs: Tensor of predicted probabilities (batch_size, sequence_length, vocabulary).
    :return: Tensor of selected token indices (batch_size, sequence_length).
    """
    return torch.argmax(pred_probs, dim=-1)



def top_k_sampling_with_scores_2d(logits, k):
    """
    Performs top-k sampling from logits, returning both sampled indices and their scores.
    Adapted for 2-dimensional logits tensor (batch_size, num_classes).

    Args:
        logits (torch.Tensor): The logits from the model (batch_size, num_classes).
        k (int): The number of top logits to consider for sampling.

    Returns:
        sampled_indices (torch.Tensor): The sampled indices (batch_size, 1).
        sampled_scores (torch.Tensor): The scores associated with the sampled indices (batch_size, 1).
    """
    # Zero out logits not in the top-k, then compute probabilities
    topk_vals, _ = torch.topk(logits, k, dim=1)
    min_vals = topk_vals[:, -1].unsqueeze(1)
    logits[logits < min_vals] = -float('Inf')
    probs = torch.softmax(logits, dim=-1)

    # Sample indices based on the modified probabilities
    sampled_indices = torch.multinomial(probs, 1)

    # Gather the scores (probabilities) corresponding to the sampled indices
    batch_indices = torch.arange(logits.size(0)).unsqueeze(1)
    sampled_scores = probs.gather(1, sampled_indices)

    return sampled_indices, sampled_scores
