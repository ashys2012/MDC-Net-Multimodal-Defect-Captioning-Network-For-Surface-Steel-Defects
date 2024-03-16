import torch


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


def giou_loss(pred_boxes, gt_boxes, no_detection_penalty=1.0):
        # Ensure both tensors are on the same device
    if pred_boxes.device != gt_boxes.device:
        gt_boxes = gt_boxes.to(pred_boxes.device)
    """
    Adjusted GIoU loss calculation to handle variable-length bounding boxes, padding, and penalize no detections.
    
    Parameters:
    - pred_boxes: Predicted bounding boxes tensor.
    - gt_boxes: Ground truth bounding boxes tensor.
    - no_detection_penalty: Fixed penalty for each ground truth box not detected.
    
    Returns:
    - Total GIoU loss with penalties for no detections.
    """
    batch_size = pred_boxes.size(0)
    giou_losses = []
    
    for i in range(batch_size):
        pred_bboxes = pred_boxes[i]
        gt_bboxes = gt_boxes[i]
        
        # Filter out zero-filled (padding) bounding boxes
        non_zero_pred = pred_bboxes.sum(dim=1) != 0
        non_zero_gt = gt_bboxes.sum(dim=1) != 0
        pred_bboxes = pred_bboxes[non_zero_pred]
        gt_bboxes = gt_bboxes[non_zero_gt]
        
        if len(pred_bboxes) == 0 and len(gt_bboxes) > 0:
            # No detections but there are ground truths; apply penalty based on the number of ground truth boxes
            giou_losses.append(torch.tensor(no_detection_penalty * len(gt_bboxes)).to(pred_boxes.device))
        elif len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
            # No valid bboxes in either predictions or ground truths, no contribution to the loss
            giou_losses.append(torch.tensor(0.0).to(pred_boxes.device))
        else:
            # Calculate GIoU for valid bounding boxes
            giou = giou_pairwise(pred_bboxes, gt_bboxes)  # Placeholder for pairwise GIoU calculation
            giou_loss = 1 - giou.mean()
            giou_losses.append(giou_loss)
    
    # Average GIoU loss across the batch
    total_giou_loss = torch.stack(giou_losses).mean()
    return total_giou_loss


def giou_pairwise(pred_boxes, gt_boxes):
    # Implement GIoU calculation here
    # This is a simplified example; replace it with your actual GIoU calculation logic
    # Ensure all tensors created or manipulated are on the correct device
    # Example: calculate a dummy GIoU tensor that respects device placement
    giou_scores = torch.rand(len(pred_boxes), len(gt_boxes), device=pred_boxes.device)
    return giou_scores

