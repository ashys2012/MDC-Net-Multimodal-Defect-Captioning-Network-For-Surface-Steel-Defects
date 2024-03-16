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
    giou_loss_meter = AvgMeter()  # For tracking IoU loss separately
    total_loss_meter = AvgMeter()  # For tracking the combined total loss
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    l1_lambda = CFG.l1_lambda  # L1 regularization strength
    iou_losses = torch.tensor([], device=CFG.device)
    no_box_penalty = .0  # Penalty for no predicted boxes

    for x, y in tqdm_object:
        x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)   
        y_input = y[:, :-1]
        y_expected = y[:, 1:]                                                   #The shape of y_expected is torch.Size([64, 99])
        y_expected_adjusted = y[:, 1:].reshape(-1)                              #The shape of y_expected_adjusted is torch.Size([6336]) (99*64)

        preds = model(x, y_input)
        preds = preds[:, :-1]                                                    #The shape of preds is torch.Size([64, 99, 305])
        predicted_classes = torch.argmax(preds, dim=-1)                          #The shape of predicted_classes is torch.Size([64, 99])


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

        print("BLEU Score:", bleu_score)

        #IoU loss calculations
        predicted_bboxes = tokenizer.decode_bboxes(tokens_caps_bbox) 
        ground_truth_bboxes = tokenizer.decode_bboxes(y_expected)

        #print("The predicted_bboxes is", predicted_bboxes)
        # print("the shape of predicted_bboxes is", predicted_bboxes.shape)
        # #print("The ground_truth_bboxes is", ground_truth_bboxes)          
        # print("the shape of ground_truth_bboxes is", ground_truth_bboxes.shape)

        iou_score = calculate_batch_iou(predicted_bboxes, ground_truth_bboxes)

        #print("The iou_score is", iou_score)

        max_ious = calculate_batch_max_iou(predicted_bboxes, ground_truth_bboxes)
        # Ensure there's at least one IoU score to avoid division by zero
        if len(max_ious) > 0:
            average_iou_score = sum(max_ious) / len(max_ious)
            print("Average IoU score:", average_iou_score)
        else:
            print("No IoU scores available to calculate average.")



        giou_bbox_loss = giou_loss(predicted_bboxes, ground_truth_bboxes)
        print("THe giou_bbox_loss is", giou_bbox_loss)



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
        ce_loss_weight = 0.3
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

def valid_epoch_bbox(model, valid_loader, criterion, tokenizer, iou_loss_individual, CFG, iou_loss_weight=0.5):
    model.eval()
    loss_meter = AvgMeter()
    iou_loss_meter = AvgMeter()
    total_loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm_object):  # Use enumerate to get an index
            x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)
            y_input = y[:, :-1]

            preds = model(x, y_input)
            preds_adjusted = preds[:, :-1, :]  # Adjust based on your model's behavior

            y_expected_adjusted = y[:, 1:].reshape(-1)

            ce_loss = criterion(preds_adjusted.reshape(-1, preds_adjusted.shape[-1]), y_expected_adjusted)

            # IOU loss calculation (omitted for brevity)

            total_loss = ce_loss  # Include other losses if necessary

            loss_meter.update(ce_loss.item(), x.size(0))
            # Update other meters similarly

            # Print statements every 10 or 15 iterations
            # if i % 10000 == 0:  # Change 10 or 15 according to your preference
            #     print(f"Iteration {i}:")
            #     print("The predictions in the valid loop are", preds)
            #     print("The y_expected_adjusted in the valid loop is", y_expected_adjusted)

            tqdm_object.set_postfix(valid_loss=total_loss_meter.avg, iou_loss=iou_loss_meter.avg)

    return loss_meter.avg, iou_loss_meter.avg, total_loss_meter.avg
