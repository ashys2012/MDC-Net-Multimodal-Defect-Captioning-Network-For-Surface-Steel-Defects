from torch import nn
import torch
import timm
from timm.models.layers import trunc_normal_
from allied_files import CFG
from utils import create_mask


from torch import nn
import torch
import timm
from timm.models.layers import trunc_normal_

class Encoder(nn.Module):
    def __init__(self, model_name='deit3_base_patch16_224', pretrained=False, out_dim=256):
        super().__init__()
        self.model = timm.create_model(
            model_name, num_classes=0, global_pool='', pretrained=pretrained)
        self.bottleneck = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x):
        features = self.model(x)
        return self.bottleneck(features[:, 1:])
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, encoder_length, dim, num_heads, num_layers):
        super().__init__()
        self.dim = dim
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, CFG.max_len-1, dim) * .02)
        self.decoder_pos_drop = nn.Dropout(p=0.4)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(dim, vocab_size)
        
        
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, encoder_length, dim) * .02)
        print("Shape of self.encoder_pos_embed:", self.encoder_pos_embed.shape)
        self.encoder_pos_drop = nn.Dropout(p=0.4)
        
        self.init_weights()
        
    def init_weights(self):
        for name, p in self.named_parameters():
            if 'encoder_pos_embed' in name or 'decoder_pos_embed' in name: 
                print("skipping pos_embed...")
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        trunc_normal_(self.encoder_pos_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)
        
    
    def forward(self, encoder_out, tgt):
        # Dynamically adjust decoder_pos_embed to match tgt length
        BOS_tesnor = torch.full((tgt.size(0), 1), CFG.bos_idx, dtype=torch.long, device=tgt.device)
        tgt_with_bos = torch.cat([BOS_tesnor, tgt], dim=1)
        sequence_length = tgt_with_bos.size(1)
        #sequence_length = tgt.size(1)
        if sequence_length != self.decoder_pos_embed.size(1):
            new_decoder_pos_embed = nn.functional.interpolate(self.decoder_pos_embed.permute(0, 2, 1), 
                                                            size=sequence_length, 
                                                            mode='linear', 
                                                            align_corners=False).permute(0, 2, 1)
        else:
            new_decoder_pos_embed = self.decoder_pos_embed

        tgt_embedding = self.embedding(tgt_with_bos)
        tgt_embedding = self.decoder_pos_drop(tgt_embedding + new_decoder_pos_embed)

        encoder_out = self.encoder_pos_drop(encoder_out + self.encoder_pos_embed)
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        # Generate autoregressive mask and padding mask for tgt
        tgt_mask, tgt_padding_mask = create_mask(tgt_with_bos)  # Corrected to pass `tgt` directly
        # print("The shape of tgt_mask is ", tgt_mask.shape)
        # print("The shape of tgt_padding_mask is ", tgt_padding_mask.shape)


        preds = self.decoder(memory=encoder_out, tgt=tgt_embedding, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        preds = preds.transpose(0, 1)

        return self.output(preds)


    
    def predict(self, encoder_out, tgt):
        length = tgt.size(1)
        padding = torch.ones(tgt.size(0), CFG.max_len-length-1).fill_(CFG.pad_idx).long().to(tgt.device)
        tgt = torch.cat([tgt, padding], dim=1)
        tgt_mask, tgt_padding_mask = create_mask(tgt)
        # is it necessary to multiply it by math.sqrt(d) ?
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )
        
        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )
        
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        
        preds = self.decoder(memory=encoder_out, 
                             tgt=tgt_embedding,
                             tgt_mask=tgt_mask, 
                             tgt_key_padding_mask=tgt_padding_mask)
        
        preds = preds.transpose(0, 1)
        output = self.output(preds)
        bos_tokens = torch.full((output.size(0), 1, output.size(2)), CFG.bos_idx, device=output.device, dtype=torch.long)
        #print("The BOS Token is --------------", bos_tokens)
    
        # Concatenate the BOS tokens to the beginning of the output
        # Since output is likely in logits or probabilities, you might directly work with indices for prepending
        # If working directly with logits or probabilities, consider a different approach to prepend BOS token
        output_with_bos = torch.cat([bos_tokens, output[:, :-1, :]], dim=1)
        
        
    
        return output_with_bos
    

# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
    
#     def forward(self, image, tgt):
#         encoder_out = self.encoder(image)
#         preds = self.decoder(encoder_out, tgt)
#         return preds
#     def predict(self, image, tgt):
#         encoder_out = self.encoder(image)
#         preds = self.decoder.predict(encoder_out, tgt)
#         return preds
    
    import torch

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, patch_dropout_rate=0.02):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.patch_dropout_rate = patch_dropout_rate  # Dropout rate for patch embeddings
    
    def forward(self, image, tgt):
        encoder_out = self.encoder(image)
        
        # Apply patch dropout to simulate masking using Uniform Random Sampling
        if self.training:  # Only during training
            batch_size, num_patches, embedding_dim = encoder_out.shape
            num_to_mask = int(num_patches * self.patch_dropout_rate)  # Calculate the number of patches to mask
            
            # Ensure at least one patch is always masked if the dropout rate is > 0
            num_to_mask = max(1, num_to_mask)
            
            # Create a mask for the patches
            mask = torch.ones((batch_size, num_patches, 1), device=encoder_out.device)
            
            for i in range(batch_size):
                indices_to_mask = torch.randperm(num_patches)[:num_to_mask]  # Randomly select patches to mask
                mask[i, indices_to_mask, :] = 0  # Apply masking
                
            encoder_out = encoder_out * mask  # Mask the encoder output
        
        preds = self.decoder(encoder_out, tgt)
        return preds
    
    def predict(self, image, tgt):
        # Ensure dropout is not applied during prediction
        encoder_out = self.encoder(image)
        preds = self.decoder.predict(encoder_out, tgt)
        return preds