import torch
from allied_files import CFG
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=CFG.device))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# def create_mask(tgt):
#     """
#     tgt: shape(N, L)
#     """
#     tgt_seq_len = tgt.shape[1]

#     tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
#     tgt_padding_mask = (tgt == CFG.pad_idx).to(tgt_mask.dtype)  # Convert to same dtype if necessary

#     return tgt_mask, tgt_padding_mask

def create_mask(tgt):
    tgt_seq_len = tgt.shape[1]  # This should reflect the sequence length with BOS token
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
    tgt_padding_mask = (tgt == CFG.pad_idx).to(tgt.device).float()  # Assuming pad_idx is defined in CFG
    return tgt_mask, tgt_padding_mask


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0]*3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

def calculate_bleu_scores(ground_truths, predictions):
    chencherry = SmoothingFunction()
    scores = []
    for ref, pred in zip(ground_truths, predictions):
        # Wrap the reference in a list as expected by sentence_bleu
        score = sentence_bleu([ref], pred, smoothing_function=chencherry.method1)
        scores.append(score)
    return scores