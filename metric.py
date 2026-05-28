import torch

def compute_iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
    # x1, y1, x2, y2 for each box
    x1 = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    y1 = torch.max(bboxes1[:, 1], bboxes2[:, 1])
    x2 = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    y2 = torch.min(bboxes1[:, 3], bboxes2[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    intersection = inter_w * inter_h

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    union = area1 + area2 - intersection

    iou = intersection / union.clamp(min=1e-6)  

    return iou


def compute_localization_metrics(gt_bb: torch.Tensor,
                                 pred_bb: torch.Tensor,
                                 iou_thresholds: list = [0.5, 0.75]) -> dict:
    """
    gt_bb:   Tensor of shape [N, 4] in (x1, y1, x2, y2) format
    pred_bb: Tensor of shape [N, 4] in (x1, y1, x2, y2) format
    """
    iou = compute_iou(gt_bb, pred_bb)

    metrics = {
        'mean_iou': iou.mean().item(),
        'median_iou': iou.median().item(),
    }

    for thresh in iou_thresholds:
        acc = (iou >= thresh).float().mean().item() * 100
        metrics[f'accuracy@{thresh}'] = acc

    return metrics


if __name__ == '__main__':
    gt_bb = torch.tensor([
        [10, 20, 100, 150],
        [30, 40, 200, 300],
        [50, 50, 180, 180],
        [0,  0,  64,  64],
        [15, 15, 85,  85]
    ], dtype=torch.float32)

    pred_bb = torch.tensor([
        [12, 18, 98,  148],   
        [30, 40, 190, 290],   
        [60, 60, 200, 200],   
        [30, 30, 94,  94],    
        [0,  0,  50,  50]     
    ], dtype=torch.float32)

    metrics = compute_localization_metrics(gt_bb, pred_bb, iou_thresholds=[0.5, 0.75])
    print(metrics)
