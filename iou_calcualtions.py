import torch

from torchvision.ops import box_iou

def bbox_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1 : tensor
        shape (N, 4), where N is the number of boxes, 4 is for [xmin, ymin, xmax, ymax]
    box2 : tensor
        shape (M, 4), where M is the number of boxes, 4 is for [xmin, ymin, xmax, ymax]
    
    Returns
    -------
    iou : tensor
        shape (N, M) IoU of boxes
    """
    # Ensure both tensors are on the same device
    if box1.device != box2.device:
        box2 = box2.to(box1.device)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    inter_xmin = torch.max(box1[:, None, 0], box2[:, 0])
    inter_ymin = torch.max(box1[:, None, 1], box2[:, 1])
    inter_xmax = torch.min(box1[:, None, 2], box2[:, 2])
    inter_ymax = torch.min(box1[:, None, 3], box2[:, 3])
    
    inter_area = (inter_xmax - inter_xmin).clamp(min=0) * (inter_ymax - inter_ymin).clamp(min=0)
    
    union_area = area1[:, None] + area2 - inter_area
    
    iou = inter_area / (union_area + 1e-6)

    
    return iou




def calculate_batch_iou(predicted_bboxes, ground_truth_bboxes):
    batch_size = predicted_bboxes.size(0)
    ious = []

    for i in range(batch_size):
        pred_boxes = predicted_bboxes[i]
        gt_boxes = ground_truth_bboxes[i]

        iou = bbox_iou(pred_boxes, gt_boxes)
        ious.append(iou)

    return ious


def calculate_batch_max_iou(predicted_bboxes, ground_truth_bboxes):
    batch_size = predicted_bboxes.size(0)
    max_ious = []

    for i in range(batch_size):
        pred_boxes = predicted_bboxes[i]  # Predicted bboxes for the current image
        gt_boxes = ground_truth_bboxes[i]  # Ground truth bboxes for the current image

        if pred_boxes.size(0) > 0 and gt_boxes.size(0) > 0:
            iou = bbox_iou(pred_boxes, gt_boxes)
            max_iou, _ = torch.max(iou, dim=1)  # Max IoU for each predicted box
            max_ious.extend(max_iou.tolist())  # Convert to list and store
        else:
            pass
            # Handle cases with no predictions or no ground truths by skipping or assigning default values

    return max_ious


def calculate_batch_max_iou_torchvision(predicted_bboxes, ground_truth_bboxes):
    batch_size = predicted_bboxes.size(0)
    max_ious = []

    # Determine the device to use based on the predicted_bboxes tensor
    device = predicted_bboxes.device

    for i in range(batch_size):
        # Remove the middle dimension for predicted_bboxes since it's 1
        pred_boxes = predicted_bboxes[i].squeeze(0).to(device)  # Ensure pred_boxes is on the correct device
        
        gt_boxes = ground_truth_bboxes[i].to(device)  # Ensure gt_boxes is on the correct device

        # Ensure pred_boxes and gt_boxes are 2-dimensional
        if pred_boxes.dim() == 1:
            pred_boxes = pred_boxes.unsqueeze(0)  # Adds an extra dimension if it's needed

        if gt_boxes.dim() == 1:
            gt_boxes = gt_boxes.unsqueeze(0)  # Adds an extra dimension if it's needed

        # Calculate IoU scores only if both sets of boxes are non-empty
        if pred_boxes.nelement() > 0 and gt_boxes.nelement() > 0:
            iou_scores = box_iou(pred_boxes, gt_boxes)
            iou_scores = torch.nan_to_num(iou_scores, nan=0.0)
            max_iou, _ = torch.max(iou_scores, dim=1)  # Max IoU for each predicted box
            max_ious.extend(max_iou.tolist())  # Convert to list and store
        #print("Iou scores inside calculate_batch_max_iou_torchvision:", iou_scores)
    return max_ious




# def giou_loss(pred_boxes, gt_boxes):


#     # Debugging: Print shapes of the input tensors
#     print("Shape of pred_boxes:", pred_boxes.shape)
#     print("Shape of gt_boxes:", gt_boxes.shape)
#     print("Immediate Shape of pred_boxes:", pred_boxes.shape)

#     area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
#     area2 = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    
#     # Debugging: Print a sample of bounding boxes to check their values
#     print("Sample pred_box:", pred_boxes[0, :])  # Print the first bounding box from pred_boxes
#     print("Sample gt_box:", gt_boxes[0, :])     # Print the first bounding box from gt_boxes

#     inter_xmin = torch.max(pred_boxes[:, None, 0], gt_boxes[:, 0])
#     inter_ymin = torch.max(pred_boxes[:, None, 1], gt_boxes[:, 1])
#     inter_xmax = torch.min(pred_boxes[:, None, 2], gt_boxes[:, 2])
#     inter_ymax = torch.min(pred_boxes[:, None, 3], gt_boxes[:, 3])

#     # Before the line causing the error, add a print statement to check values
#     print("inter_xmin shape:", inter_xmin.shape)
#     print("inter_ymin shape:", inter_ymin.shape)
#     print("Attempting to calculate inter_xmax...")
#     inter_area = (inter_xmax - inter_xmin).clamp(min=0) * (inter_ymax - inter_ymin).clamp(min=0)

#     # Calculate union
#     pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
#     gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
#     union_area = pred_area + gt_area - inter_area

#     # Calculate IoU
#     iou = inter_area / (union_area + 1e-6)

#     # Calculate the smallest enclosing box
#     enc_xmin = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
#     enc_ymin = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
#     enc_xmax = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
#     enc_ymax = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
#     enc_area = (enc_xmax - enc_xmin) * (enc_ymax - enc_ymin)

#     # Calculate GIoU
#     giou = iou - (enc_area - union_area) / (enc_area + 1e-6)
#     loss = 1 - giou  # GIoU loss

#     return loss.mean()

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from allied_files import CFG

device = CFG.device 


def giou_loss_with_scores(pred_boxes, gt_boxes, no_detection_penalty=1.0):
    """
    Adjusted GIoU loss calculation to handle variable-length bounding boxes, padding, penalize no detections,
    and also return GIoU scores.

    Parameters:
    - pred_boxes: Predicted bounding boxes tensor.
    - gt_boxes: Ground truth bounding boxes tensor.
    - no_detection_penalty: Fixed penalty for each ground truth box not detected.
    
    Returns:
    - Total GIoU loss with penalties for no detections.
    - A list of GIoU scores for each batch item.
    """
    if pred_boxes.device != gt_boxes.device:
        gt_boxes = gt_boxes.to(pred_boxes.device)

    batch_size = pred_boxes.size(0)
    giou_losses = []
    giou_scores_batch = []

    for i in range(batch_size):
        pred_bboxes = pred_boxes[i]
        gt_bboxes = gt_boxes[i]

        non_zero_pred = pred_bboxes.sum(dim=1) != 0
        non_zero_gt = gt_bboxes.sum(dim=1) != 0
        pred_bboxes = pred_bboxes[non_zero_pred]
        gt_bboxes = gt_bboxes[non_zero_gt]

        if len(pred_bboxes) == 0 and len(gt_bboxes) > 0:
            giou_losses.append(torch.tensor(no_detection_penalty * len(gt_bboxes)).to(pred_boxes.device))
            giou_scores_batch.append(torch.tensor([]).to(pred_boxes.device))
        elif len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
            giou_losses.append(torch.tensor(0.0).to(pred_boxes.device))
            giou_scores_batch.append(torch.tensor([]).to(pred_boxes.device))
        else:
            giou = giou_pairwise(pred_bboxes, gt_bboxes)
            giou_loss = 1 - giou.mean()
            giou_losses.append(giou_loss)
            giou_scores_batch.append(giou)

    total_giou_loss = torch.stack(giou_losses).mean()
    return total_giou_loss, giou_scores_batch



# def giou_pairwise(pred_boxes, gt_boxes):
#     # Implement GIoU calculation here
#     # This is a simplified example; replace it with your actual GIoU calculation logic
#     # Ensure all tensors created or manipulated are on the correct device
#     # Example: calculate a dummy GIoU tensor that respects device placement
#     giou_scores = torch.rand(len(pred_boxes), len(gt_boxes), device=pred_boxes.device)
#     return giou_scores

def giou_pairwise(pred_boxes, gt_boxes):
    """
    Calculate the Generalized Intersection over Union (GIoU) between two sets of boxes.
    
    Args:
    - pred_boxes (Tensor): Predicted bounding boxes, shape (N, 4).
    - gt_boxes (Tensor): Ground truth bounding boxes, shape (M, 4).
    
    Returns:
    - Tensor: GIoU scores, shape (N, M).
    """
    # Intersection
    max_xy = torch.min(pred_boxes[:, None, 2:], gt_boxes[:, 2:])
    min_xy = torch.max(pred_boxes[:, None, :2], gt_boxes[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    intersection = inter[:, :, 0] * inter[:, :, 1]
    
    # Areas
    pred_boxes_area = ((pred_boxes[:, 2] - pred_boxes[:, 0]) * 
                       (pred_boxes[:, 3] - pred_boxes[:, 1]))
    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * 
                     (gt_boxes[:, 3] - gt_boxes[:, 1]))
    
    # Union
    union = pred_boxes_area[:, None] + gt_boxes_area[None, :] - intersection
    
    # Enclosing box
    enc_max_xy = torch.max(pred_boxes[:, None, 2:], gt_boxes[:, 2:])
    enc_min_xy = torch.min(pred_boxes[:, None, :2], gt_boxes[:, :2])
    enc = enc_max_xy - enc_min_xy
    enclosing_area = enc[:, :, 0] * enc[:, :, 1]
    
    # GIoU
    iou = intersection / union
    giou = iou - (enclosing_area - union) / enclosing_area
    return giou



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