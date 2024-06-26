import torch
from tqdm import tqdm
from allied_files import CFG, AvgMeter, get_lr
from iou_bbox import decode_bbox_from_pred, decode_predictions,decode_single_prediction,extract_ground_truth,iou_loss, extract_predictions,  calculate_iou, iou_loss_individual
from data_processing import Tokenizer, Vocabulary, top_k_sampling, extract_tokens, top_k_sampling_with_scores_2d
#torch.set_printoptions(profile="full")
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from iou_calcualtions import bbox_iou, calculate_batch_iou, calculate_batch_max_iou, giou_loss_with_scores, calculate_batch_max_iou_torchvision
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pandas as pd
from datetime import datetime
from utilities import append_df_to_csv, draw_bbox_with_caption
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils import calculate_bleu_scores
from collections import defaultdict
import math

vocab = Vocabulary(freq_threshold=5)
tokenizer = Tokenizer(vocab, num_classes=10, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
CFG.bos_idx = tokenizer.BOS_code  
CFG.pad_idx = tokenizer.PAD_code



def train_epoch(model, train_loader, optimizer, lr_scheduler, criterion, logger=None, iou_loss_weight = CFG.iou_loss_weight):
    model.train()
    loss_meter = AvgMeter()
    giou_loss_meter = AvgMeter()  # For tracking IoU loss separately
    total_loss_meter = AvgMeter()  # For tracking the combined total loss
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    l1_lambda = CFG.l1_lambda  # L1 regularization strength
    iou_losses = torch.tensor([], device=CFG.device)
    no_box_penalty = .0  # Penalty for no predicted boxes
    epoch_captions_preds_list = []
    epoch_caption_grnd_truth_list = []
    preds_epoch_bbox = []
    targets_epoch_bbox = []
    iou_meter = AvgMeter("Average IoU")
    giou_loss_meter = AvgMeter("GIoU Loss")



    for x, y in tqdm_object:
        x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)   
        y_input = y[:, :-1]
        y_expected = y[:, 1:]                                                   #The shape of y_expected is torch.Size([64, 99])
        #print("The shape of y_expected is", y_expected.shape)
        y_expected_adjusted = y[:, 1:].reshape(-1)                              #The shape of y_expected_adjusted is torch.Size([6336]) (99*64)
        #print("The shape of y_expected_adjusted is", y_expected_adjusted.shape)

        preds = model(x, y_input)
        preds = preds[:, :-1]                                                   # The forced BOS tesnor is removed hence this is commented and the shape of preds is torch.Size([64, 99, 305])
        predicted_classes = torch.argmax(preds, dim=-1)                          #The shape of predicted_classes is torch.Size([64, 99])
        #print(f"Preds shape: {preds.reshape(-1, preds.shape[-1]).shape}")



        # #Label cross entropy loss calculations
        # predicted_labels = tokenizer.extract_predicted_labels_with_logits(preds) #The shape of predicted_labels is torch.Size([64, 305]) --[batch_size, num_classes]
        # ground_truth_labels = tokenizer.decode_labels(y_expected)                #The shape of ground_truth_labels is torch.Size([64])
        # predicted_labels = predicted_labels.float()
        # Label_loss = criterion(predicted_labels, ground_truth_labels)            # CFG.pad_idx is the PAD_TOKEN in your decode_labels function

        # #Classification Accuracy calculations#
        # final_predicted_classes = tokenizer.decode_labels(predicted_classes)     #The shape of  final predicted classes value is torch.Size([64])
        # # Accuracy calculation might need to account for PAD_TOKENs
        # valid_indices = final_predicted_classes != CFG.pad_idx                   # Ignore PAD_TOKEN in accuracy calculation
        # train_accuracy_with_no_penalty = (final_predicted_classes[valid_indices] == ground_truth_labels[valid_indices]).float().mean()
        # train_accuracy = (final_predicted_classes == ground_truth_labels) & valid_indices
        # train_accuracy_with_penalty = train_accuracy.float().mean()              #This accuracy even counts if ther is no predictions of labels
        # To print the accuracy you need to use item() at the end.

        #BLEU score calculations for captions
        tokens_caps_bbox = top_k_sampling(preds.reshape(-1, preds.size(-1)), k=5).reshape(preds.size(0), -1)
        caption_grnd_truth = tokenizer.decode_captions(y_expected)             #The shape of caption_grnd_truth is torch.Size([64, 99]

        captions_preds = tokenizer.decode_captions(tokens_caps_bbox)             #The shape of captions_preds is torch.Size([64, 99])
        # print("The shape of captions_preds is", captions_preds.shape)
        # print("The captions_preds is", captions_preds)
        # print("The shape of caption_grnd_truth is", caption_grnd_truth.shape)
        # print("The caption_grnd_truth is", caption_grnd_truth)
        # Convert tensors to lists of integers (token IDs)
        captions_preds_list = captions_preds.cpu().tolist()
        caption_grnd_truth_list = [caption_grnd_truth.cpu().tolist()]  # Note: ground truth should be a list of references, hence the extra []

        # Calculate BLEU score
        # We use a list of lists for the ground truth to simulate multiple reference translations, which is the common format for BLEU calculation.
        # Even though we have only one reference, it needs to be wrapped in another list.
        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu(caption_grnd_truth_list, captions_preds_list, 
                           smoothing_function=chencherry.method1)

        # if logger is not None:
        #     logger.log({"Normal BLEU Score training is": bleu_score})




        #ALternate bleu score calculations
        # Convert the ground truth and predictions to text
        # Define the stoi and itos dictionaries as provided
        stoi = {
            '<PAD>': 302, '<SOS>': 300, '<EOS>': 301, '<UNK>': 299, 'oil_spot': 262, 'inclusion': 264, 'crescent_gap': 260, 
            'water_spot': 261, 'punching_hole': 258, 'welding_line': 259, 'silk_spot': 263, 'rolled_pit': 265, 'crease': 266, 
            'waist_folding': 267, 'the': 270, 'defect': 271, 'is': 272, 'located': 273, 'at': 274, 'center': 275, 
            'of': 276, 'image': 277, '.': 278, 'right': 279, 'bottom': 280, 'top': 281, 'left': 282
}

        itos = {
            302: '<PAD>', 300: '<SOS>', 301: '<EOS>', 299: '<UNK>', 262: 'oil_spot', 264: 'inclusion', 260: 'crescent_gap', 
            261: 'water_spot', 258: 'punching_hole', 259: 'welding_line', 263: 'silk_spot', 265: 'rolled_pit', 266: 'crease', 
            267: 'waist_folding', 270: 'the', 271: 'defect', 272: 'is', 273: 'located', 274: 'at', 275: 'center', 
            276: 'of', 277: 'image', 278: '.', 279: 'right', 280: 'bottom', 281: 'top', 282: 'left'
        }
        # Reverse the stoi dictionary to be able to use ids to get tokens
        itos = defaultdict(lambda: '<UNK>', itos)    

        ground_truth_text = [tokenizer.tokens_to_text_new(caption, itos) for caption in caption_grnd_truth_list]
        predictions_text = tokenizer.tokens_to_text_new(captions_preds_list, itos)

        # Calculate BLEU scores
        bleu_scores_trial = calculate_bleu_scores(ground_truth_text[0], predictions_text)  # Only one ground truth set in this case
        if logger is not None:
            logger.log({"BLEU Score for training with text is": bleu_scores_trial})


        # Below code gives an error called unhashable list
        
        #Epoch wise Bleu Score avg calculation
        # Inside your iteration loop, after obtaining captions_preds_list and caption_grnd_truth_list
        epoch_captions_preds_list.extend(predictions_text)
        epoch_caption_grnd_truth_list.extend(ground_truth_text[0])  # Assuming caption_grnd_truth_list is wrapped in an additional list

        # Make sure the ground truth list is in the correct format for BLEU calculation
        #epoch_caption_grnd_truth_list = [epoch_caption_grnd_truth_list]  # Wrap in another list

        # Compute BLEU score for the entire epoch
        chencherry = SmoothingFunction()
        bleu_score_epoch = sentence_bleu(epoch_caption_grnd_truth_list, epoch_captions_preds_list, smoothing_function=chencherry.method1)

        #No proper output in WandB

        # # Log the epoch BLEU score
        # if logger is not None:
        #     logger.log({"Epoch BLEU Score": bleu_score_epoch})




        #IoU loss calculations
        predicted_bboxes = tokenizer.decode_bboxes(tokens_caps_bbox) 
        #print("---------#################################---------------------")
        ground_truth_bboxes = tokenizer.decode_bboxes(y_expected)     #The shape of ground_truth_bboxes is torch.Size([32, 5, 4]) the center value is variable as it is the num_bboxes
    



        #Mean Average Precision calculations


        # Reshape logits for sampling if necessary
        reshaped_logits = preds.reshape(-1, preds.size(-1))    #The shape of reshaped_logits is torch.Size([3168, 305])

        # Perform top-k sampling to get indices and scores
        sampled_indices, sampled_scores = top_k_sampling_with_scores_2d(reshaped_logits, k=10)

        # If needed, reshape back to original dimensions
        # Be mindful of how you reshape indices and scores to align with your downstream processing
        sampled_indices = sampled_indices.reshape(preds.size(0), -1)            #The shape of sampled_indices is torch.Size([32, 99]) =3168
        sampled_scores = sampled_scores.reshape(preds.size(0), -1)              #The shape of sampled_scores is torch.Size([3168, 1])

        bbox_w_score, label_w_score, score_ = tokenizer.decode_bboxes_and_labels_with_scores(sampled_indices, sampled_scores)
        #The shape of bbox_w_score is torch.Size([32, 1, 4]) and the shape of label_w_score is torch.Size([32, 1])
        ground_truth_bboxes, ground_truth_labels = tokenizer.decode_bboxes_and_labels(y_expected)

        preds_formatted = []
        targets_formatted = []
        actual_batch_size = bbox_w_score.shape[0]

        for i in range(actual_batch_size):  # Assuming 32 is your batch size
            # Formatting preds
            pred_dict = {
                "boxes": bbox_w_score[i],  # Assuming shape [1, 4]
                "scores": score_[i],       # Assuming shape [1]
                "labels": label_w_score[i] # Assuming shape [1]
            }
            preds_formatted.append(pred_dict)
            
            # Formatting target
            # This part depends on how you have your ground truth labels organized
            # Let's assume you have a similar tensor for labels: ground_truth_labels with shape [32, 5]
            target_dict = {
                "boxes": ground_truth_bboxes[i],  # Assuming shape [5, 4]
                "labels": ground_truth_labels[i]  # Assuming shape [5]
            }
            targets_formatted.append(target_dict)

        # Step 2: Initialize the mAP metric
        # Adjust parameters according to your specific needs, for example, setting different IoU thresholds
        # Example of specifying IoU thresholds as a list
        iou_thresholds = [0.3]  # This can be a list of thresholds if needed

        map_metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=iou_thresholds, class_metrics=False)

        # Step 3: Update the metric with your predictions and targets
        # Assuming preds and targets_formatted are your prediction and ground truth data
        # Since your data is already on CUDA, ensure torchmetrics is also using the same device
        map_metric = map_metric.to(CFG.device)



        for pred, target in zip(preds_formatted, targets_formatted):
            if pred['scores'].nelement() == 0:  # Checks if there are no scores
                pred = {
                    "boxes": torch.empty((0, 4), device=CFG.device),
                    "scores": torch.empty((0,), device=CFG.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=CFG.device)
                }
            map_metric.update([pred], [target])



        # Step 4: Compute the mAP
        map_scores = map_metric.compute()
        #map_scores_scalar = {k: v.item() if hasattr(v, "item") else v for k, v in map_scores.items()}

        # Extract class-wise mAP scores, assuming they're stored under "map_per_class"
        # if "map_per_class" in map_scores:
        #     class_maps = map_scores["map_per_class"].cpu().numpy()  # Ensure it's on CPU and convert to numpy for easy handling

        #     # Prepare class-wise mAP scores for logging
        #     class_maps_dict = {f"Class_{i}_mAP": mAP for i, mAP in enumerate(class_maps)}
            
        #     # Log class-wise mAP scores
        #     if logger is not None:
        #         logger.log(class_maps_dict)

        # Log overall mAP scores # COmmented because we have the avg scores now
        # if logger is not None:
        #     logger.log({"Training mAP scores": map_scores})


        #Map calculation per epoch starts here

        # Inside your batch processing loop, replace the direct mAP update with accumulation
        preds_epoch_bbox.extend(preds_formatted)
        targets_epoch_bbox.extend(targets_formatted)

        


        #print("The predicted_bboxes is", predicted_bboxes)
        #print("the shape of predicted_bboxes is", predicted_bboxes.shape)
        #print("The ground_truth_bboxes is", ground_truth_bboxes)          
        #print("the shape of ground_truth_bboxes is", ground_truth_bboxes.shape)

        iou_score = calculate_batch_iou(predicted_bboxes, ground_truth_bboxes)

        #print("The iou_score is", iou_score)

        #max_ious = calculate_batch_max_iou(predicted_bboxes, ground_truth_bboxes)
        max_ious = calculate_batch_max_iou_torchvision(predicted_bboxes, ground_truth_bboxes)
        #print("Max IOUs:", max_ious)

        # Ensure there's at least one IoU score to avoid division by zero
        # if len(max_ious) > 0:
        #     average_iou_score = sum(max_ious) / len(max_ious)
        #     print("Average IoU score:", average_iou_score)
        # else:
        #     print("No IoU scores available to calculate average.")

        if len(max_ious) > 0:
            average_iou_score = sum(max_ious) / len(max_ious)
        else:
            average_iou_score = float('nan')  # Consider how you want to handle this case for your overall epoch average

        # Update the IoU AvgMeter with the batch's average IoU score
        # If average_iou_score is 'nan', it may be skipped or handled specially
        if not math.isnan(average_iou_score):
            iou_meter.update(average_iou_score, len(predicted_bboxes))

        giou_bbox_loss, giou_bbox_score_batch = giou_loss_with_scores(predicted_bboxes, ground_truth_bboxes)

        # Update the GIoU Loss AvgMeter with the batch's GIoU loss
        giou_loss_meter.update(giou_bbox_loss.item(), len(predicted_bboxes))

        # print("THe giou_bbox_loss is", giou_bbox_loss)
        # print("THe giou_bbox_score is", giou_bbox_score)
        # if logger is not None:
        #     logger.log({
        #         "Trainig GIoU BBox Loss": giou_bbox_loss,
        #         #"Trainig GIoU BBox Score Batch": giou_bbox_score_batch,
        #         #"Trainig Length of GIoU BBox Score Batch": len(giou_bbox_score_batch)
        #         })



        decoded_lables,decoded_pred_bboxes, decoded_captions = extract_predictions(preds, tokenizer)  # Adapt this to match your actual decoding function
        # print("The decoded preditions of bbox is ", decoded_pred_bboxes)
        # print("The decoded preditions of labels is ", decoded_lables)
        # print("The decoded preditions of captions is ", decoded_captions)
        #first_batch_preds = probs[0]  # This selects the first batch
        #print("The first batch preds is", first_batch_preds)
        
        # Extract the ground truth bounding boxes from `y` or another source as needed
        _, gt_bboxes, _ = extract_ground_truth(y_expected_adjusted, tokenizer)  # You need to implement this based on your dataset


        #print("The length of decoded_pred_bboxes is", len(decoded_pred_bboxes))
        #print("THe decoded_pred_bboxes is", decoded_pred_bboxes)
        if all(len(bbox) == 0 for bbox in decoded_pred_bboxes):
            # Set iou_losses to a tensor of no_box_penalty values
            iou_losses = torch.full((len(gt_bboxes),), no_box_penalty, device=CFG.device)
            #print(f"No predicted bounding boxes. IoU losses: {iou_losses}")
        else:
            for pred_bboxes, gt_bbox in zip(decoded_pred_bboxes, gt_bboxes):
                # Ensure pred_bboxes and gt_bbox are tensors, and move them to the correct device
                pred_bboxes_tensor = torch.tensor(pred_bboxes, dtype=torch.float, device=CFG.device)
                gt_bbox_tensor = torch.tensor(gt_bbox, dtype=torch.float, device=CFG.device)

                # Calculate IoU loss for this image
                iou_loss_for_image = iou_loss_individual(pred_bboxes_tensor, gt_bbox_tensor)
                #print(f"IoU loss for image: {iou_loss_for_image}")
                iou_loss_for_image = iou_loss_for_image.to(CFG.device)
                print("The shape of iou_loss_for_image is", iou_loss_for_image.shape)
                print("THe shape of iou_losses is", iou_losses.shape)
                iou_losses = torch.cat((iou_losses, iou_loss_for_image.unsqueeze(0)), dim=0)

        # Compute mean IoU loss across the batch
        if iou_losses.nelement() > 0:  # Check if iou_losses is not empty
            iou_loss_val = iou_losses.mean()
        else:
            iou_loss_val = torch.tensor(0.0, device=CFG.device)

        #print(f"Mean IoU loss: {iou_loss_val}")
        #iou_loss_val = torch.stack(iou_losses).mean()

        #iou_loss_val = sum(iou_losses) / len(iou_losses) if iou_losses else torch.tensor(0.0, device=CFG.device)   #item gives an error as it is a float not a tensor
        #iou_loss_val = iou_loss(decoded_pred_bboxes_tensor, gt_bboxes_tensor)    #torch.stack_error code

        
        # Calculate your existing loss (e.g., cross-entropy for captions)
        #ce_loss = criterion(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
            

        ce_loss = criterion(preds.reshape(-1, preds.shape[-1]), y_expected_adjusted)






        # Calculate L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        
        # Combine losses
        ce_loss_weight = 1 - CFG.iou_loss_weight
        total_loss = ce_loss_weight * ce_loss + l1_lambda * l1_norm + iou_loss_weight * giou_bbox_loss


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update loss meters
        loss_meter.update(ce_loss.item(), x.size(0))
        #iou_loss_meter.update(iou_loss_val.item(), x.size(0))
        giou_loss_meter.update(giou_bbox_loss.item(), x.size(0))
        total_loss_meter.update(total_loss.item(), x.size(0))

        lr = get_lr(optimizer)
        tqdm_object.set_postfix(train_loss=total_loss_meter.avg, iou_loss=giou_loss_meter.avg, lr=f"{lr:.6f}")
        if logger is not None:
            logger.log({"train_step_loss": total_loss_meter.avg, "iou_loss": giou_loss_meter.avg, 'lr': lr})


    # After all batches in the epoch are processed:
    # Reinitialize the mAP metric to ensure it's reset for each epoch calculation
    map_metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=iou_thresholds, class_metrics=True)
    map_metric = map_metric.to(CFG.device)

    # Update the mAP metric with all collected predictions and targets of the epoch
    for pred, target in zip(preds_epoch_bbox, targets_epoch_bbox):
        if pred['scores'].nelement() == 0:  # Handle cases with no predictions
            pred = {
                "boxes": torch.empty((0, 4), device=CFG.device),
                "scores": torch.empty((0,), device=CFG.device),
                "labels": torch.empty((0,), dtype=torch.int64, device=CFG.device)
            }
        map_metric.update([pred], [target])

    # Compute and log the mAP for the entire epoch
    map_scores_epoch = map_metric.compute()

    # Log the epoch-level mAP to your logger (e.g., wandb)
    if logger is not None:
        logger.log({"Epoch mAP for Train": map_scores_epoch})


    if logger is not None:
        logger.log({
        "Training Average IoU Score": iou_meter.avg,
        "Training GIoU BBox Loss": giou_loss_meter.avg
         })


    

    
    return total_loss_meter.avg


# Log hyperparameters
# wandb.config.update({
#     "max_len": CFG.max_len,
#     "img_size": CFG.img_size,
#     "batch_size": CFG.batch_size,
#     "epochs": CFG.epochs,
#     "model_name": CFG.model_name,
#     "lr": CFG.lr,
#     "weight_decay": CFG.weight_decay
# })

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def valid_epoch_bbox(model, valid_loader, criterion, tokenizer, iou_loss_weight=0.5, logger=None, epoch_num=None):
    model.eval()
    loss_meter = AvgMeter()
    giou_loss_meter = AvgMeter()  # For tracking IoU loss separately
    total_loss_meter = AvgMeter()  # For tracking the combined total loss
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    log_df = pd.DataFrame()  # Initialize logging dataframe
    # Inside your iteration loop, after obtaining captions_preds_list and caption_grnd_truth_list
    epoch_captions_preds_list = []
    epoch_caption_grnd_truth_list = []
    preds_epoch_bbox = []
    targets_epoch_bbox = []
    iou_meter = AvgMeter("Average IoU")
    giou_loss_meter = AvgMeter("GIoU Loss")



    
    with torch.no_grad():
        for x, y in tqdm_object:
            x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)
            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            y_expected_adjusted = y[:, 1:].reshape(-1)

            preds = model(x, y_input)
            preds = preds[:, :-1]
            predicted_classes = torch.argmax(preds, dim=-1)

            # BLEU score calculations for captions
            tokens_caps_bbox = top_k_sampling(preds.reshape(-1, preds.size(-1)), k=5).reshape(preds.size(0), -1)
            caption_grnd_truth = tokenizer.decode_captions(y_expected)

            captions_preds = tokenizer.decode_captions(tokens_caps_bbox)
            captions_preds_list = captions_preds.cpu().tolist()
            caption_grnd_truth_list = [caption_grnd_truth.cpu().tolist()]

            chencherry = SmoothingFunction()
            bleu_score = sentence_bleu(caption_grnd_truth_list, captions_preds_list, smoothing_function=chencherry.method1)

            # Log BLEU score
            if logger is not None:
                logger.log({"Validation BLEU Score": bleu_score})






            #ALternate bleu score calculations
            # Convert the ground truth and predictions to text
            # Define the stoi and itos dictionaries as provided


            stoi = {
                '<PAD>': 302, '<SOS>': 300, '<EOS>': 301, '<UNK>': 299, 'oil_spot': 262, 'inclusion': 264, 'crescent_gap': 260, 
                'water_spot': 261, 'punching_hole': 258, 'welding_line': 259, 'silk_spot': 263, 'rolled_pit': 265, 'crease': 266, 
                'waist_folding': 267, 'the': 270, 'defect': 271, 'is': 272, 'located': 273, 'at': 274, 'center': 275, 
                'of': 276, 'image': 277, '.': 278, 'right': 279, 'bottom': 280, 'top': 281, 'left': 282
    }

            itos = {
                302: '<PAD>', 300: '<SOS>', 301: '<EOS>', 299: '<UNK>', 262: 'oil_spot', 264: 'inclusion', 260: 'crescent_gap', 
                261: 'water_spot', 258: 'punching_hole', 259: 'welding_line', 263: 'silk_spot', 265: 'rolled_pit', 266: 'crease', 
                267: 'waist_folding', 270: 'the', 271: 'defect', 272: 'is', 273: 'located', 274: 'at', 275: 'center', 
                276: 'of', 277: 'image', 278: '.', 279: 'right', 280: 'bottom', 281: 'top', 282: 'left'
            }






            # Reverse the stoi dictionary to be able to use ids to get tokens
            itos = defaultdict(lambda: '<UNK>', itos)    

            ground_truth_text = [tokenizer.tokens_to_text_new(caption, itos) for caption in caption_grnd_truth_list]
            predictions_text = tokenizer.tokens_to_text_new(captions_preds_list, itos)

            # Calculate BLEU scores
            bleu_scores_trial = calculate_bleu_scores(ground_truth_text[0], predictions_text)  # Only one ground truth set in this case
            if logger is not None:
                logger.log({"BLEU Score for validation with text is": bleu_scores_trial})


            # Below code gives an error called unhashable list
                                #Epoch wise Bleu Score avg calculation
            # Inside your iteration loop, after obtaining captions_preds_list and caption_grnd_truth_list
            epoch_captions_preds_list.extend(predictions_text)
            epoch_caption_grnd_truth_list.extend(ground_truth_text[0])  # Assuming caption_grnd_truth_list is wrapped in an additional list

            # Make sure the ground truth list is in the correct format for BLEU calculation
            #epoch_caption_grnd_truth_list = [epoch_caption_grnd_truth_list]  # Wrap in another list

            # Compute BLEU score for the entire epoch
            chencherry = SmoothingFunction()
            bleu_score_epoch = sentence_bleu(epoch_caption_grnd_truth_list, epoch_captions_preds_list, smoothing_function=chencherry.method1)

            # Log the epoch BLEU score
            if logger is not None:
                logger.log({"Epoch BLEU Score": bleu_score_epoch})





        

            # IoU and GIoU loss calculations
            predicted_bboxes = tokenizer.decode_bboxes(tokens_caps_bbox)
            ground_truth_bboxes = tokenizer.decode_bboxes(y_expected)

            iou_score = calculate_batch_iou(predicted_bboxes, ground_truth_bboxes)
            max_ious = calculate_batch_max_iou(predicted_bboxes, ground_truth_bboxes)



            if len(max_ious) > 0:
                average_iou_score = sum(max_ious) / len(max_ious)
            else:
                average_iou_score = float('nan')  # Consider how you want to handle this case for your overall epoch average

            # Update the IoU AvgMeter with the batch's average IoU score
            # If average_iou_score is 'nan', it may be skipped or handled specially
            if not math.isnan(average_iou_score):
                iou_meter.update(average_iou_score, len(predicted_bboxes))

            giou_bbox_loss, giou_bbox_score_batch = giou_loss_with_scores(predicted_bboxes, ground_truth_bboxes)

            # Update the GIoU Loss AvgMeter with the batch's GIoU loss
            giou_loss_meter.update(giou_bbox_loss.item(), len(predicted_bboxes))
            
            # if len(max_ious) > 0:
            #     average_iou_score = sum(max_ious) / len(max_ious)
            #     # Log average IoU score
            #     if logger is not None:
            #         logger.log({"Validation Average IoU score": average_iou_score})

            # giou_bbox_loss, giou_bbox_score_batch = giou_loss_with_scores(predicted_bboxes, ground_truth_bboxes)
            # # Log GIoU bbox loss and score batch
            # if logger is not None:
            #     logger.log({
            #         "Validation GIoU BBox Loss": giou_bbox_loss,
            #         "Validation GIoU BBox Score Batch": giou_bbox_score_batch,
            #         "Validation Length of GIoU BBox Score Batch": len(giou_bbox_score_batch)
            #     })





            #Mean Average Precision calculations


            # Reshape logits for sampling if necessary
            reshaped_logits = preds.reshape(-1, preds.size(-1))    #The shape of reshaped_logits is torch.Size([3168, 305])

            # Perform top-k sampling to get indices and scores
            sampled_indices, sampled_scores = top_k_sampling_with_scores_2d(reshaped_logits, k=10)

            # If needed, reshape back to original dimensions
            # Be mindful of how you reshape indices and scores to align with your downstream processing
            sampled_indices = sampled_indices.reshape(preds.size(0), -1)            #The shape of sampled_indices is torch.Size([32, 99]) =3168
            sampled_scores = sampled_scores.reshape(preds.size(0), -1)              #The shape of sampled_scores is torch.Size([3168, 1])

            bbox_w_score, label_w_score, score_ = tokenizer.decode_bboxes_and_labels_with_scores(sampled_indices, sampled_scores)
            #The shape of bbox_w_score is torch.Size([32, 1, 4]) and the shape of label_w_score is torch.Size([32, 1])
            ground_truth_bboxes, ground_truth_labels = tokenizer.decode_bboxes_and_labels(y_expected)

            preds_formatted = []
            targets_formatted = []
            actual_batch_size = bbox_w_score.shape[0]

            for i in range(actual_batch_size):  # Assuming 32 is your batch size
                # Formatting preds
                pred_dict = {
                    "boxes": bbox_w_score[i],  # Assuming shape [1, 4]
                    "scores": score_[i],       # Assuming shape [1]
                    "labels": label_w_score[i] # Assuming shape [1]
                }
                preds_formatted.append(pred_dict)
                
                # Formatting target
                # This part depends on how you have your ground truth labels organized
                # Let's assume you have a similar tensor for labels: ground_truth_labels with shape [32, 5]
                target_dict = {
                    "boxes": ground_truth_bboxes[i],  # Assuming shape [5, 4]
                    "labels": ground_truth_labels[i]  # Assuming shape [5]
                }
                targets_formatted.append(target_dict)

            # Step 2: Initialize the mAP metric
            # Adjust parameters according to your specific needs, for example, setting different IoU thresholds
            # Example of specifying IoU thresholds as a list
            iou_thresholds = [0.3]  # This can be a list of thresholds if needed

            map_metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=iou_thresholds, class_metrics=True)

            # Step 3: Update the metric with your predictions and targets
            # Assuming preds and targets_formatted are your prediction and ground truth data
            # Since your data is already on CUDA, ensure torchmetrics is also using the same device
            map_metric = map_metric.to(CFG.device)



            for pred, target in zip(preds_formatted, targets_formatted):
                if pred['scores'].nelement() == 0:  # Checks if there are no scores
                    pred = {
                        "boxes": torch.empty((0, 4), device=CFG.device),
                        "scores": torch.empty((0,), device=CFG.device),
                        "labels": torch.empty((0,), dtype=torch.int64, device=CFG.device)
                    }
                map_metric.update([pred], [target])



            # Step 4: Compute the mAP
            map_scores = map_metric.compute()
            #map_scores_scalar = {k: v.item() if hasattr(v, "item") else v for k, v in map_scores.items()}

            # Extract class-wise mAP scores, assuming they're stored under "map_per_class"
            if "map_per_class" in map_scores:
                class_maps = map_scores["map_per_class"].cpu().numpy()  # Ensure it's on CPU and convert to numpy for easy handling

                # Prepare class-wise mAP scores for logging
                class_maps_dict = {f"Class_{i}_mAP": mAP for i, mAP in enumerate(class_maps)}
                
                # Log class-wise mAP scores
                if logger is not None:
                    logger.log(class_maps_dict)

            # Log overall mAP scores
            if logger is not None:
                logger.log({"Validation mAP scores": map_scores})


            # Inside your batch processing loop, replace the direct mAP update with accumulation
            preds_epoch_bbox.extend(preds_formatted)
            targets_epoch_bbox.extend(targets_formatted)



            ce_loss = criterion(preds.reshape(-1, preds.shape[-1]), y_expected_adjusted)
            # Note: L1 regularization is not applied during validation

            total_loss = ce_loss + iou_loss_weight * giou_bbox_loss  # Combine losses

            # Update loss meters
            loss_meter.update(ce_loss.item(), x.size(0))
            giou_loss_meter.update(giou_bbox_loss.item(), x.size(0))
            total_loss_meter.update(total_loss.item(), x.size(0))

            tqdm_object.set_postfix(valid_loss=total_loss_meter.avg, giou_loss=giou_loss_meter.avg)

        # After all batches in the epoch are processed:
        # Reinitialize the mAP metric to ensure it's reset for each epoch calculation
        map_metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=iou_thresholds, class_metrics=True)
        map_metric = map_metric.to(CFG.device)

        # Update the mAP metric with all collected predictions and targets of the epoch
        for pred, target in zip(preds_epoch_bbox, targets_epoch_bbox):
            if pred['scores'].nelement() == 0:  # Handle cases with no predictions
                pred = {
                    "boxes": torch.empty((0, 4), device=CFG.device),
                    "scores": torch.empty((0,), device=CFG.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=CFG.device)
                }
            map_metric.update([pred], [target])

        # Compute and log the mAP for the entire epoch
        map_scores_epoch = map_metric.compute()

        # Log the epoch-level mAP to your logger (e.g., wandb)
        if logger is not None:
            logger.log({"Epoch mAP for Validation": map_scores_epoch})

        
        if logger is not None:
            logger.log({
                "Validation Average IoU Score": iou_meter.avg,
                "Validation GIoU BBox Loss": giou_loss_meter.avg
            })





    return loss_meter.avg, giou_loss_meter.avg, total_loss_meter.avg



def test_epoch(model, test_loader, tokenizer, save_dir='/mnt/sdb/2024/pix_2_seq_with_captions_GC_10_dataset/test_output_images',logger = None, epoch_num=None):
    model.eval()
    batch_counter = 0  # Initialize batch counter
    log_data = []
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(CFG.device)
            #print("The shape of x is", x.shape)
            # Assuming y contains both the captions and bounding boxes information
            y_input = y[:, :-1].to(CFG.device)
            y_expected = y[:, 1:]

            # Generate predictions
            preds = model.predict(x, y_input)[:, :-1]
            predicted_classes = torch.argmax(preds, dim=-1)
            
            # Decode predictions to get captions and bounding boxes
            predicted_captions = tokenizer.decode_captions(torch.argmax(preds, dim=-1))



            tokens_caps_bbox = top_k_sampling(preds.reshape(-1, preds.size(-1)), k=5).reshape(preds.size(0), -1)
            predicted_bboxes = tokenizer.decode_bboxes(tokens_caps_bbox)  # Ensure this function exists and works as expected

            # BLEU score calculations for captions
            tokens_caps_bbox = top_k_sampling(preds.reshape(-1, preds.size(-1)), k=5).reshape(preds.size(0), -1)
            caption_grnd_truth = tokenizer.decode_captions(y_expected)

            captions_preds = tokenizer.decode_captions(tokens_caps_bbox)
            captions_preds_list = captions_preds.cpu().tolist()
            caption_grnd_truth_list = [caption_grnd_truth.cpu().tolist()]

            chencherry = SmoothingFunction()
            bleu_score = sentence_bleu(caption_grnd_truth_list, captions_preds_list, smoothing_function=chencherry.method1)

            # Log BLEU score
            if logger is not None:
                logger.log({"Testing BLEU Score": bleu_score})





            #ALternate bleu score calculations
            # Convert the ground truth and predictions to text
            # Define the stoi and itos dictionaries as provided


            stoi = {
                '<PAD>': 302, '<SOS>': 300, '<EOS>': 301, '<UNK>': 299, 'oil_spot': 262, 'inclusion': 264, 'crescent_gap': 260, 
                'water_spot': 261, 'punching_hole': 258, 'welding_line': 259, 'silk_spot': 263, 'rolled_pit': 265, 'crease': 266, 
                'waist_folding': 267, 'the': 270, 'defect': 271, 'is': 272, 'located': 273, 'at': 274, 'center': 275, 
                'of': 276, 'image': 277, '.': 278, 'right': 279, 'bottom': 280, 'top': 281, 'left': 282
    }

            itos = {
                302: '<PAD>', 300: '<SOS>', 301: '<EOS>', 299: '<UNK>', 262: 'oil_spot', 264: 'inclusion', 260: 'crescent_gap', 
                261: 'water_spot', 258: 'punching_hole', 259: 'welding_line', 263: 'silk_spot', 265: 'rolled_pit', 266: 'crease', 
                267: 'waist_folding', 270: 'the', 271: 'defect', 272: 'is', 273: 'located', 274: 'at', 275: 'center', 
                276: 'of', 277: 'image', 278: '.', 279: 'right', 280: 'bottom', 281: 'top', 282: 'left'
            }






            # Reverse the stoi dictionary to be able to use ids to get tokens
            itos = defaultdict(lambda: '<UNK>', itos)    

            ground_truth_text = [tokenizer.tokens_to_text_new(caption, itos) for caption in caption_grnd_truth_list]
            predictions_text = tokenizer.tokens_to_text_new(captions_preds_list, itos)

            # Calculate BLEU scores
            bleu_scores_trial = calculate_bleu_scores(ground_truth_text[0], predictions_text)  # Only one ground truth set in this case
            if logger is not None:
                logger.log({"BLEU Score for Testing with text is": bleu_scores_trial})


            
            # IoU and GIoU loss calculations
            predicted_bboxes = tokenizer.decode_bboxes(tokens_caps_bbox)
            ground_truth_bboxes = tokenizer.decode_bboxes(y_expected)

            iou_score = calculate_batch_iou(predicted_bboxes, ground_truth_bboxes)
            max_ious = calculate_batch_max_iou(predicted_bboxes, ground_truth_bboxes)
            
            if len(max_ious) > 0:
                average_iou_score = sum(max_ious) / len(max_ious)
                # Log average IoU score
                if logger is not None:
                    logger.log({"Testing Average IoU score": average_iou_score})

            giou_bbox_loss, giou_bbox_score_batch = giou_loss_with_scores(predicted_bboxes, ground_truth_bboxes)
            # Log GIoU bbox loss and score batch
            if logger is not None:
                logger.log({
                    "Testing GIoU BBox Loss": giou_bbox_loss,
                    "Testing GIoU BBox Score Batch": giou_bbox_score_batch,
                    #"Testing Length of GIoU BBox Score Batch": len(giou_bbox_score_batch)
                })

            
            batch_counter += 1

            log_temp = {
            'epoch_batch': f"Epoch_{epoch_num}_Batch_{batch_counter}",
            'captions_preds_list': [predictions_text],
            'caption_grnd_truth_list': [ground_truth_text],
            'predicted_bboxes': [predicted_bboxes.tolist()],  # Assuming predicted_bboxes is a tensor
            'ground_truth_bboxes': [ground_truth_bboxes.tolist()],  # Assuming ground_truth_bboxes is a tensor
            'predicted_classes': [predicted_classes.tolist()],  # Assuming predicted_classes is a tensor
            'y_expected': [y_expected.tolist()]  # Assuming y_expected is a tensor
            }
            log_data.append(log_temp)  # Append to log_data list

        

            # After completing the loop for all batches in an epoch
            log_df = pd.DataFrame(log_data)  # Convert your accumulated log data into a DataFrame

            # Append this epoch's log DataFrame to the Excel file
            output_file_path = f"/mnt/sdb/2024/pix_2_seq_with_captions_GC_10_dataset/output_excel_file_results_validation/epoch_map/validation_log_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
            append_df_to_csv(output_file_path, log_df)




            
            # Convert batch of tensors to numpy images and process each image
            # Convert batch of tensors to numpy images and process each image
            # images_np = x.cpu().numpy()
            # num_items = min(x.size(0), predicted_captions.size(0))
            # for i in range(num_items):  # Iterate over the range of num_items instead
            #     image_np = images_np[i]

            #     # Ensure the image data is in the range [0, 255] if it was normalized
            #     image_np = (image_np * 255).astype(np.uint8)

            #     # Rearrange the array from (C, H, W) to (H, W, C) for PIL compatibility
            #     image_np = np.transpose(image_np, (1, 2, 0))

            #     # Now convert to a PIL Image
            #     image_pil = Image.fromarray(image_np).convert('RGB')

            #     # Example of adjusting the call for bboxes and captions
            #     bboxes = [list(map(int, bbox)) for bbox in predicted_bboxes[i].tolist()] if predicted_bboxes[i].dim() > 0 else []
            #     #print("The bboxes is in the test epoch is", bboxes)
            #     captions = predicted_captions[i].tolist() if predicted_captions[i].dim() > 0 else []
            #     #print("The captions is in the test epoch is", captions)

            #     # Now, bboxes and captions are both lists, and we can safely attempt to draw them
            #     draw_bbox_with_caption(image_pil, bboxes, captions)

            #     # Save the image with unique naming including epoch number
            #     filename = f'test_image_epoch{epoch_num}_batch{batch_idx}_img{i}.png'
            #     image_pil.save(os.path.join(save_dir, filename))