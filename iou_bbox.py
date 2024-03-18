import torch

def calculate_iou(pred_boxes, gt_boxes):
    """
    Calculate intersection over union for pairs of bounding boxes.
    Assumes boxes are in [x_min, y_min, x_max, y_max] format.
    
    Args:
        pred_boxes (Tensor): Predicted bounding boxes, shape [N, 4].
        gt_boxes (Tensor): Ground truth bounding boxes, shape [M, 4].
        
    Returns:
        Tensor: IoU scores for each pair of boxes.
    """
    # Ensure pred_boxes and gt_boxes are not empty and have the correct shape
    if pred_boxes.nelement() == 0 or gt_boxes.nelement() == 0:
        # Handle empty inputs case; adjust based on your application needs
        return torch.tensor(0.0) 

    # Ensure both pred_boxes and gt_boxes are 2D
    if pred_boxes.dim() < 2:
        pred_boxes = pred_boxes.unsqueeze(0)
    if gt_boxes.dim() < 2:
        gt_boxes = gt_boxes.unsqueeze(0)

    # Calculate intersection
    inter_xmin = torch.max(pred_boxes[:, 0].unsqueeze(1), gt_boxes[:, 0])
    inter_ymin = torch.max(pred_boxes[:, 1].unsqueeze(1), gt_boxes[:, 1])
    inter_xmax = torch.min(pred_boxes[:, 2].unsqueeze(1), gt_boxes[:, 2])
    inter_ymax = torch.min(pred_boxes[:, 3].unsqueeze(1), gt_boxes[:, 3])

    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)
    
    # Calculate union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    
    union_area = pred_area.unsqueeze(1) + gt_area - inter_area  # Adjust broadcasting for pred_area

    # Calculate IoU
    iou = inter_area / union_area
    
    return iou


def iou_loss(pred_boxes, gt_boxes, min_penalty=0.5):
    """
    Calculate IoU loss for pairs of bounding boxes, applying a minimum penalty for zero IoU.
    
    Args:
        pred_boxes (Tensor): Predicted bounding boxes, shape [N, 4].
        gt_boxes (Tensor): Ground truth bounding boxes, shape [N, 4].
        min_penalty (float): Minimum penalty to apply when IoU is zero.
        
    Returns:
        Tensor: Mean IoU loss with minimum penalty applied.
    """
    ious = calculate_iou(pred_boxes, gt_boxes)
    # Apply minimum penalty for zero IoU values
    ious = torch.where(ious > 0, ious, torch.tensor(min_penalty, device=ious.device))
    # Calculate loss as 1 - IoU, applying the minimum penalty for zero IoU cases
    iou_loss = 1 - ious
    return iou_loss.mean()



def calculate_iou_individual(pred_box, gt_boxes):
    """
    Calculate IoU for a single predicted box against multiple ground truth boxes.
    
    Args:
        pred_box (Tensor): Single predicted bounding box, shape [1, 4].
        gt_boxes (Tensor): Ground truth bounding boxes, shape [N, 4].
        
    Returns:
        Tensor: IoUs for the predicted box against each ground truth box, shape [N].
    """
    # Assuming calculate_iou can handle broadcasting or has been adapted to
    # calculate IoUs for one pred_box against multiple gt_boxes
    ious = calculate_iou(pred_box.expand_as(gt_boxes), gt_boxes)
    return ious



def iou_loss_individual(pred_boxes, gt_boxes, min_penalty=0.1, no_box_penalty=1.0):
    """
    Calculate IoU loss for individual pairs of predicted and ground truth boxes, applying a minimum penalty.
    
    Args:
        pred_boxes (Tensor): Predicted bounding boxes, shape [N, 4].
        gt_boxes (Tensor): Ground truth bounding boxes, shape [N, 4].
        min_penalty (float): Minimum penalty to apply when IoU is zero.
        no_box_penalty (float): Penalty to apply when no boxes are predicted.
        
    Returns:
        Tensor: IoU loss for each predicted box, with minimum penalty applied, shape [N].
    """
    if pred_boxes.nelement() == 0:
        # Return the no_box_penalty as the loss if no predicted boxes
        return torch.full((gt_boxes.size(0),), no_box_penalty, device=gt_boxes.device)

    iou_losses = []
    for pred_box in pred_boxes:
        ious = calculate_iou_individual(pred_box.unsqueeze(0), gt_boxes)
        print(f"IoUs: {ious}")
        # Apply minimum penalty for zero IoU values
        ious = torch.where(ious > 0, ious, torch.full_like(ious, min_penalty))
        # Calculate loss as 1 - IoU for each pair, applying minimum penalty
        loss = 1 - ious
        iou_losses.append(loss)

    # Stack to create a tensor of individual losses and calculate mean loss per predicted box
    iou_losses = torch.stack(iou_losses).mean(dim=1)
    return iou_losses.mean()




def extract_ground_truth(token_sequences, tokenizer):
    """
    Utilize the tokenizer's decode function to extract labels, bounding boxes, and captions
    from a batch of token sequences representing the ground truth.

    Args:
        token_sequences (Tensor): Batch of token sequences, shape [batch_size, sequence_length].
        tokenizer (Tokenizer): Initialized tokenizer with a decode method.

    Returns:
        Three lists containing labels, bounding boxes, and captions for each sequence in the batch.
    """
    all_labels = []
    all_bboxes = []
    all_captions = []
    #print("The ground truth token sequnces in extract gnd truth is", token_sequences)
    #print("------Here is the ground truth tokenizer decoded-------------")
    for tokens in token_sequences:
        labels, bboxes, caption = tokenizer.decode(tokens)
        all_labels.append(labels)
        all_bboxes.append(bboxes)
        all_captions.append(caption)
#     print("The gnd truth of all_labels",all_labels)
#     print("The gnd truth of all_bboxes",all_bboxes)
#     print("The gnd truth of all_captions",all_captions)

    return all_labels, all_bboxes, all_captions



def extract_predictions(token_sequences, tokenizer):
    """
    Utilize the tokenizer's decode function to extract labels, bounding boxes, and captions
    from a batch of token sequences representing the ground truth.

    Args:
        token_sequences (Tensor): Batch of token sequences, shape [batch_size, sequence_length].
        tokenizer (Tokenizer): Initialized tokenizer with a decode method.

    Returns:
        Three lists containing labels, bounding boxes, and captions for each sequence in the batch.
    """
    all_labels = []
    all_bboxes = []
    all_captions = []
    #print("The ground truth token sequnces in extract gnd truth is", token_sequences)
    #print("------Here is the ground truth tokenizer decoded-------------")
    for tokens in token_sequences:
        labels, bboxes, caption = tokenizer.decode(tokens)
        all_labels.append(labels)
        all_bboxes.append(bboxes)
        all_captions.append(caption)
#     print("The gnd truth of all_labels",all_labels)
#     print("The gnd truth of all_bboxes",all_bboxes)
#     print("The gnd truth of all_captions",all_captions)

    return all_labels, all_bboxes, all_captions




def decode_bbox_from_pred(preds, tokenizer):
    """
    Decode bounding box predictions from model's output.
    
    Args:
        preds (Tensor): Raw predictions from the model, shape [batch_size, sequence_length, vocab_size].
        tokenizer (Tokenizer): The tokenizer instance used for decoding.
        
    Returns:
        List of decoded labels and bounding boxes for each item in the batch.
    """
    # Convert logits to token indices (e.g., using argmax over the vocabulary dimension)
    token_indices = preds.argmax(dim=-1)  # Shape: [batch_size, sequence_length]
    
    decoded_labels_batch = []
    decoded_bboxes_batch = []
    decoded_captions_batch = []
    
    for token_sequence in token_indices:
        # Ensure token_sequence is a tensor here. In this context, it already is, so no conversion is needed.
        # Decode the sequence of token indices into labels, bounding boxes, and captions.
        # Ensure your tokenizer's decode method can handle tensor input directly.
        decoded_labels, decoded_bboxes, decoded_caption = tokenizer.decode(token_sequence)
        decoded_labels_batch.append(decoded_labels)
        decoded_bboxes_batch.append(decoded_bboxes)
        decoded_captions_batch.append(decoded_caption)
    
    return decoded_labels_batch, decoded_bboxes_batch, decoded_captions_batch


def decode_predictions(preds, tokenizer):
    #print("The preds in the decode prediction are", preds)
    decoded_bboxes = []  # Placeholder for decoded bounding boxes
    decoded_labels = []  # Placeholder for decoded labels
    decoded_captions = []  # Placeholder for decoded captions

    # Ensure preds is in an iterable format (batch processing)
    # No need to check for 0-dimensionality here since preds is expected to be a batch of predictions
    if preds.dim() == 1:
        # This handles a case where preds is a 1D tensor, suggesting single sequence processing
        # Wrap it in a list to simulate batch processing with one element
        preds = preds.unsqueeze(0)
    
    # Iterate over batch
    for pred in preds:
        # Assuming 'pred' is now a single prediction tensor
        # Apply your decoding logic here
        bbox, label, caption = decode_single_prediction(pred, tokenizer)  # Adapt this to your actual single prediction decoding function

        decoded_bboxes.append(bbox)
        decoded_labels.append(label)
        decoded_captions.append(caption)
    
#     print("Decoded Labels in preds decode_predictions:", decoded_labels)
#     print("Decoded preds BBoxes inside decode_predictions:", decoded_bboxes)
#     print("Decoded Caption in preds decode_predictions:", decoded_captions)

    return decoded_labels, decoded_bboxes, decoded_captions

def decode_single_prediction(pred, tokenizer):
    # Placeholder function for decoding a single prediction tensor
    # You need to implement this logic based on your specific model output and decoding needs
    #print("THe predictions in the decode_single_prediction are ",pred)
   # print("------Here is the prediction tokenizer decoded-------------")
    bbox, label, caption = tokenizer.decode(pred)
    bbox = []  # Dummy decoded bounding box
    label = []  # Dummy decoded label
    caption = ""  # Dummy decoded caption
    # Implement actual decoding logic here
    
    return bbox, label, caption