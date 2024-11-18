import torch

def get_msp(model, data):
    logits = model(data)
    probs = torch.softmax(logits, dim=1)
    scores, _ = torch.max(probs, dim=1)
    return scores