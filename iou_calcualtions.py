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
