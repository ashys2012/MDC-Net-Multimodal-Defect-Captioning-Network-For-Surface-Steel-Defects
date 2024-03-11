import numpy as np
from collections import Counter
import spacy
import torch
from sklearn.model_selection import train_test_split
from dataset import get_transform_train, get_transform_valid, collate_fn
import os
from functools import partial
import cv2
spacy_eng = spacy.load('en_core_web_sm') 

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {302: "<PAD>", 300: "<SOS>", 301: "<EOS>", 299: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

        # Predefined indices for specific words
        self.predefined_indices = {
            'patches': 262,
            'inclusion': 263,
            'rolled-in_scale': 260,
            'pitted_surface': 261,
            'crazing': 258,
            'scratches': 259
        }

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        compound_words = ['rolled-in_scale', 'pitted_surface', 'crazing', 'scratches']

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






class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None, tokenizer=None):
        self.ids = df['ids'].unique()
        self.df = df
        self.transforms = transforms
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sample = self.df[self.df['ids'] == self.ids[idx]]
        #print("The sample is:", sample)
        img_path = sample['img_path'].values[0]
        if not os.path.exists(img_path):
            print(f"Warning: The file '{img_path}' does not exist. Skipping.")
            return None

        img = cv2.imread(img_path)[..., ::-1]  # Convert BGR to RGB

        concatenated = sample['concatenated'].values[0]  # Assuming one entry per 'ids'
        labels = [item[0] for item in concatenated]  # Extracting labels from 'concatenated'
        #print("labels in Vocdataset gives you", labels)
        bboxes = [item[1:5] for item in concatenated]  # Extracting bbox coordinates
        #print("bboxes in Vocdataset gives you", bboxes)
        caption = sample['caption'].values[0]  # Assuming one caption per 'ids'
        #print("caption in Vocdataset gives you", caption)
        

        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=bboxes, labels=labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
            #print("bboxes transformed", bboxes)
        img = torch.FloatTensor(img).permute(2, 0, 1)  # Convert to torch tensor and permute to CxHxW

        if self.tokenizer is not None:
            seqs = self.tokenizer(labels, bboxes, [caption])  # Tokenizer expects a list of captions
            seqs = torch.LongTensor(seqs)
            #print("the seq in voc dataset is ", seqs)
            return img, seqs

        # If the tokenizer isn't provided, you might want to return the raw data or handle this case differently
        return img, labels, bboxes, caption

    def __len__(self):
        return len(self.ids)


def get_loaders(df, tokenizer, img_size, batch_size, max_len, pad_idx, num_workers=2, valid_size=0.2):
    
    # Split the original data into train and validation dataframes
    train_df, valid_df = train_test_split(df, test_size=valid_size, random_state=42)

    print("training dataset is ", len(train_df))
    print("val dataset is ", len(valid_df))
    
    train_ds = VOCDataset(train_df, transforms=get_transform_train(img_size), tokenizer=tokenizer)
    valid_ds = VOCDataset(valid_df, transforms=get_transform_valid(img_size), tokenizer=tokenizer)
    
    #THe below code is to check the dataset with its bbox
    #
    num_samples_to_print = 1

    for i in range(num_samples_to_print):
        # Get the ith sample
        img_tensors, bbox_label_seq = train_ds[i]  # Unpack the tuple here
        
        # Print the bounding boxes and labels
        print(f"Sample {i}:")
        print("Img_tensors:", img_tensors)
        print("Seq:", bbox_label_seq)
        print("\n")


    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    # print("Train size: ", train_df['id'].nunique())
    # print("Valid size: ", valid_df['id'].nunique())
        
    return train_loader, valid_loader  # Removed test_loader



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

    def __call__(self, labels, bboxes, captions, shuffle=False):
        #print("the original bboxes ", bboxes)
        bboxes = np.array(bboxes, dtype=float)
        #print("tha bboxes after they are converted to np array", bboxes)
        labels = np.array(labels).astype('int')[:self.max_len]
        #print("tha labels after they are converted to np array", labels)
#         print("self.width is", self.width)
#         print("self.height is", self.height)

        # Normalize bounding box coordinates and quantize
        bboxes[:, 0] = bboxes[:, 0] / self.width
        bboxes[:, 2] = bboxes[:, 2] / self.width
        bboxes[:, 1] = bboxes[:, 1] / self.height
        bboxes[:, 3] = bboxes[:, 3] / self.height
        
        #print("self.width is-------->",self.width )
        
#         print("Max bbox value before quantization:", np.max(bboxes))
#         #print("Number of classes:", self.num_classes)
#         print("The bbox values before quantization",bboxes)
        bboxes = self.quantize(bboxes)[:self.max_len] #+ self.num_classes
#         print("The bbox values after quantization",bboxes)
        
#         print("Max bbox value after the procss of quantizatioin", np.max(bboxes))

        if shuffle:
            rand_idxs = np.arange(len(bboxes))
            np.random.shuffle(rand_idxs)
            labels = labels[rand_idxs]
            bboxes = bboxes[rand_idxs]
            # Do not apply rand_idxs to captions since there's only one caption per image

        tokenized = [self.BOS_code]
        
        # Tokenize the single caption assuming all captions are identical for shuffled bboxes
        if not isinstance(captions, list):
            captions = [captions] 
        caption = captions[0]  # Take the first (and only) caption for tokenization
        caption_tokens = [self.CAPTION_START]
        caption_tokens.extend(self.vocab.numericalize(caption))
        tokenized.extend(caption_tokens)
        tokenized.append(self.CAPTION_END)  # Mark the end of the caption
        
        
        for label, bbox in zip(labels, bboxes):
            # First append the label
            tokenized.append(label)

            # Then extend the tokenized list with the bbox coordinates
            tokenized.extend(map(int, bbox))



        tokenized.append(self.EOS_code)

        return tokenized[:self.max_len]

    
    def get_spacy_vocab_size(self):
        return len(self.vocab)

    def get_complete_vocab_size(self):
        return self.vocab_size

    
    def print_vocab(self):
        print("STOI:", self.vocab.stoi)
        print("ITOS:", self.vocab.itos)
    
    # def decode(self, tokens: torch.tensor):
    #     """
    #     tokens: torch.LongTensor with shape [L]
    #     """
    #     mask = tokens != self.PAD_code
    #     tokens = tokens[mask]
    #     print(type(tokens), tokens)
    #     print("Tokens:-----", tokens)
    #     tokens = tokens[1:-1]
        
        
        
    #     # Assuming each object is represented by 5 tokens: 4 for bbox and 1 for label
    #     # and each caption is represented by N tokens
    #     assert len(tokens) % (5 + self.caption_length) == 0, "invalid tokens"
    def decode(self, tokens):
        #tokens = torch.tensor(tokens)  # Convert list to PyTorch tensor commented due to warning of unecssary memory allocation
        #tokens = tokens.clone().detach()
        
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)  # Adjust dtype as necessary
        # Now tokens is guaranteed to be a tensor, so we can clssone and detach
        tokens = tokens.clone().detach()
        
           
        

        #print("The tokens from tokenizer is before removing EOS and BOS", tokens)

        if tokens.numel() == 0:  # Check if tensor is empty
            return [], [], []

        # Remove BOS and EOS tokens
        #tokens = tokens[1:-1]
        #tokens = tokens[(tokens != self.PAD_code)]
        
       # print("The tokens from tokenizer is after removing EOS and BOS", tokens)
        #print("The data type of tokens is:", type(tokens))
        #print("The length of tokens is:", len(tokens))

        labels = []
        bboxes = []
        caption_tokens = []
        #print("Raw tokens from the decode func of tokenizer",tokens)
        # Find SOC (Start of Caption) and EOC (End of Caption) indices
        soc_idxs = (tokens == self.CAPTION_START).nonzero(as_tuple=True)[0]
       # print("The soc Idx are ----", soc_idxs)
        if soc_idxs.nelement() > 0:
            soc_idx = soc_idxs[0].item()  # Use the first occurrence
        else:
            # Handle the case where CAPTION_START is not found
            soc_idx = None

        # Apply a similar check for CAPTION_END
        eoc_idxs = (tokens == self.CAPTION_END).nonzero(as_tuple=True)[0]
        if eoc_idxs.nelement() > 0:
            eoc_idx = eoc_idxs[0].item()  # Use the first occurrence
        else:
            # Handle the case where CAPTION_END is not found
            eoc_idx = None

        # Proceed with decoding only if both start and end indices are found
        if soc_idx is not None and eoc_idx is not None:
            # Extract and process the caption using soc_idx and eoc_idx
            # Remember to adjust subsequent logic to account for the possibility
            # that soc_idx or eoc_idx could be None
            caption_tokens = tokens[soc_idx+1:eoc_idx]
        else:
            # Handle cases where SOC or EOC are not found appropriately
            pass
            #print("SOC or EOC token not found.")


        # Decode Caption
        #caption_tokens = tokens[soc_idx+1:eoc_idx]
        #captions_text = self.tokens_to_text([caption_tokens.tolist()])

        # Assuming the previous checks for soc_idx and eoc_idx
        # and setting them to None if not found...

        if soc_idx is not None and eoc_idx is not None:
            # It's safe to proceed with operations involving soc_idx and eoc_idx
            caption_tokens = tokens[soc_idx+1:eoc_idx]
            captions_text = self.tokens_to_text([caption_tokens.tolist()])

            # Process BBoxes and Labels after EOC
            bbox_label_tokens = tokens[eoc_idx+1:]
            for i in range(0, len(bbox_label_tokens), 5):  # Assuming format [label, xmin, ymin, xmax, ymax]
                # Ensure i+4 does not exceed bbox_label_tokens length
                if i+4 < len(bbox_label_tokens):
                    label_tensor = bbox_label_tokens[i:i+1]  # Extract as tensor
                    if label_tensor.numel() == 1:  # Ensure it's a single-element tensor
                        label = label_tensor.item()  # Convert to Python scalar
                        labels.append(int(label))
                        bbox = bbox_label_tokens[i+1:i+5]
                        bboxes.append([int(item) for item in bbox])
                    else:
                        # Handle unexpected tensor size (e.g., log an error or throw an exception)
                        print(f"Unexpected label_tensor size: {label_tensor}")
        else:
            # Handle cases where SOC or EOC are not found appropriately
           # print("SOC or EOC token not found.")
            # Optionally set default values or perform cleanup actions here
            captions_text = ""
            labels = []
            bboxes = []


        # Adjust BBoxes to original dimensions
        bboxes = np.array(bboxes, dtype=float)
        # Check if 'bboxes' is empty
        if bboxes.size > 0:
            if bboxes.ndim == 1:
                # Proceed if it's a 1D array and non-empty
                bboxes[0] = self.dequantize(bboxes[0]) * self.width
                bboxes[2] = self.dequantize(bboxes[2]) * self.width
                # And similarly for y coordinates
                bboxes[1] = self.dequantize(bboxes[1]) * self.height
                bboxes[3] = self.dequantize(bboxes[3]) * self.height
                #print("bboxes in the decode are bboxes.size > 0:",bboxes)
            elif bboxes.ndim == 2:
                # If 'bboxes' is 2D, you might handle it differently, for example:
                bboxes[:, [0, 2]] = self.dequantize(bboxes[:, [0, 2]]) * self.width
                bboxes[:, [1, 3]] = self.dequantize(bboxes[:, [1, 3]]) * self.height
                #print("bboxes in the decode are  bboxes.ndim == 2:",bboxes)
                
        else:
            # Handle the case where 'bboxes' is empty
           # print("Warning: No bounding boxes to decode.")
            # Optionally, set 'bboxes' to a default value or handle as appropriate
            pass


        return labels, bboxes, captions_text[0] if captions_text else ""


    

    
    def tokens_to_text(self, captions):
        # Convert list of token lists back to list of caption texts
        return [" ".join([self.vocab.itos.get(token, '<UNK>') for token in caption]) for caption in captions]