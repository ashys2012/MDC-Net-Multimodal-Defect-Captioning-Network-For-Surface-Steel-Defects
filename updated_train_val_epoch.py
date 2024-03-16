import torch
from tqdm import tqdm
from allied_files import CFG, AvgMeter, get_lr
from iou_bbox import decode_bbox_from_pred, decode_predictions,decode_single_prediction,extract_ground_truth,iou_loss, extract_predictions,  calculate_iou, iou_loss_individual
from data_processing import Tokenizer, Vocabulary, top_k_sampling, extract_tokens
#torch.set_printoptions(profile="full")
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from iou_calcualtions import bbox_iou, calculate_batch_iou, calculate_batch_max_iou,giou_loss
vocab = Vocabulary(freq_threshold=5)
tokenizer = Tokenizer(vocab, num_classes=6, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
CFG.bos_idx = tokenizer.BOS_code
CFG.pad_idx = tokenizer.PAD_code


def train_epoch(model, train_loader, optimizer, lr_scheduler, criterion, logger=None, iou_loss_weight=0.7):
    model.train()
    loss_meter = AvgMeter()
    giou_loss_meter = AvgMeter()  
    total_loss_meter = AvgMeter()  
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    l1_lambda = CFG.l1_lambda  
    iou_losses = torch.tensor([], device=CFG.device)
    no_box_penalty = .0  

    for x, y in tqdm_object:
        x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)   
        y_input = y[:, :-1]
        y_expected = y[:, 1:]                                                   
        y_expected_adjusted = y[:, 1:].reshape(-1)                              

        preds = model(x, y_input)
        preds = preds[:, :-1]                                                    
        predicted_classes = torch.argmax(preds, dim=-1)                          

        tokens_caps_bbox = top_k_sampling(preds.reshape(-1, preds.size(-1)), k=5).reshape(preds.size(0), -1)
        caption_grnd_truth = tokenizer.decode_captions(y_expected)             

        captions_preds = tokenizer.decode_captions(tokens_caps_bbox)             
        captions_preds_list = captions_preds.cpu().tolist()
        caption_grnd_truth_list = [caption_grnd_truth.cpu().tolist()]  

        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu(caption_grnd_truth_list, captions_preds_list, 
                                   smoothing_function=chencherry.method1)

        predicted_bboxes = tokenizer.decode_bboxes(tokens_caps_bbox) 
        ground_truth_bboxes = tokenizer.decode_bboxes(y_expected)

        iou_score = calculate_batch_iou(predicted_bboxes, ground_truth_bboxes)

        max_ious = calculate_batch_max_iou(predicted_bboxes, ground_truth_bboxes)
        if len(max_ious) > 0:
            average_iou_score = sum(max_ious) / len(max_ious)
        else:
            print("No IoU scores available to calculate average.")

        giou_bbox_loss = giou_loss(predicted_bboxes, ground_truth_bboxes)

        decoded_lables, decoded_pred_bboxes, decoded_captions = extract_predictions(preds, tokenizer)  
        _, gt_bboxes, _ = extract_ground_truth(y_expected_adjusted, tokenizer)  

        if all(len(bbox) == 0 for bbox in decoded_pred_bboxes):
            iou_losses = torch.full((len(gt_bboxes),), no_box_penalty, device=CFG.device)
        else:
            for pred_bboxes, gt_bbox in zip(decoded_pred_bboxes, gt_bboxes):
                pred_bboxes_tensor = torch.tensor(pred_bboxes, dtype=torch.float, device=CFG.device)
                gt_bbox_tensor = torch.tensor(gt_bbox, dtype=torch.float, device=CFG.device)
                iou_loss_for_image = iou_loss_individual(pred_bboxes_tensor, gt_bbox_tensor)
                iou_losses = torch.cat((iou_losses, iou_loss_for_image.unsqueeze(0)), dim=0)

        if iou_losses.nelement() > 0:
            iou_loss_val = iou_losses.mean()
        else:
            iou_loss_val = torch.tensor(0.0, device=CFG.device)

        ce_loss = criterion(preds.reshape(-1, preds.shape[-1]), y_expected_adjusted)

        l1_norm = sum(p.abs().sum() for p in model.parameters())
        
        ce_loss_weight = 0.3
        total_loss = ce_loss_weight * ce_loss + l1_lambda * l1_norm + iou_loss_weight * giou_bbox_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_meter.update(ce_loss.item(), x.size(0))
        giou_loss_meter.update(giou_bbox_loss.item(), x.size(0))
        total_loss_meter.update(total_loss.item(), x.size(0))

        lr = get_lr(optimizer)
        tqdm_object.set_postfix(train_loss=total_loss_meter.avg, iou_loss=giou_loss_meter.avg, lr=f"{lr:.6f}")
        if logger is not None:
            logger.log({"train_step_loss": total_loss_meter.avg, "iou_loss": giou_loss_meter.avg, 'lr': lr})
    
    return total_loss_meter.avg



def valid_epoch_bbox(model, valid_loader, criterion, tokenizer, iou_loss_individual, CFG, iou_loss_weight=0.5):
    model.eval()
    loss_meter = AvgMeter()
    iou_loss_meter = AvgMeter()
    total_loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm_object):
            x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)
            y_input = y[:, :-1]

            preds = model(x, y_input)
            preds_adjusted = preds[:, :-1, :]
            y_expected_adjusted = y[:, 1:].reshape(-1)

            ce_loss = criterion(preds_adjusted.reshape(-1, preds_adjusted.shape[-1]), y_expected_adjusted)

            # Generate bounding boxes and captions for validation data
            tokens_caps_bbox = top_k_sampling(preds_adjusted.reshape(-1, preds_adjusted.size(-1)), k=5).reshape(preds_adjusted.size(0), -1)
            caption_grnd_truth = tokenizer.decode_captions(y[:, 1:])  # Adjust according to your data structure
            captions_preds = tokenizer.decode_captions(tokens_caps_bbox)

            predicted_bboxes = tokenizer.decode_bboxes(tokens_caps_bbox) 
            ground_truth_bboxes = tokenizer.decode_bboxes(y[:, 1:])  # Adjust according to your data structure

            # Calculate IoU or GIoU scores here as done in training
            giou_bbox_loss = giou_loss(predicted_bboxes, ground_truth_bboxes)

            # If you wish to calculate and include IoU loss
            # iou_loss_val = calculate_iou_loss_somehow(predicted_bboxes, ground_truth_bboxes)

            total_loss = ce_loss + iou_loss_weight * giou_bbox_loss  # Adjust formula if necessary

            loss_meter.update(ce_loss.item(), x.size(0))
            iou_loss_meter.update(giou_bbox_loss.item(), x.size(0))  # If IoU loss was calculated
            total_loss_meter.update(total_loss.item(), x.size(0))

            tqdm_object.set_postfix(valid_loss=total_loss_meter.avg, iou_loss=iou_loss_meter.avg)

    return loss_meter.avg, iou_loss_meter.avg, total_loss_meter.avg
