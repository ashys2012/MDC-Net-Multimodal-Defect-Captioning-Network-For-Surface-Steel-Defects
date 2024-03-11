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


def iou_loss_individual(pred_boxes, gt_boxes, min_penalty=0.1):
    # Check if pred_boxes is empty properly
    # If pred_boxes is a list, check if the list is empty
    # If pred_boxes is a tensor, check if it has any elements
    is_pred_boxes_empty = (isinstance(pred_boxes, list) and len(pred_boxes) == 0) or \
                          (isinstance(pred_boxes, torch.Tensor) and pred_boxes.nelement() == 0)

    # If pred_boxes is empty, return the minimum penalty as the IoU loss
    if is_pred_boxes_empty:
        # Ensure to return on the correct device, similar to gt_boxes if available, or default to 'cpu'
        device = gt_boxes.device if isinstance(gt_boxes, torch.Tensor) and gt_boxes.nelement() > 0 else torch.device('cpu')
        return torch.tensor(min_penalty, device=device)

    # Continue with your existing logic if pred_boxes is not empty
    ious = []
    for index, pred_box in enumerate(pred_boxes):
        ious_for_pred = [calculate_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)) for gt_box in gt_boxes]
        best_iou = max(ious_for_pred) if ious_for_pred else torch.tensor(min_penalty, device=pred_boxes[0].device)
        best_iou = torch.where(best_iou > 0, best_iou, torch.tensor(min_penalty, device=pred_boxes[0].device))
        ious.append(best_iou)

    average_iou = torch.stack(ious).mean() if ious else torch.tensor(min_penalty, device=pred_boxes[0].device)
    iou_loss = 1 - average_iou
    return iou_loss


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
