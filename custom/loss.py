import torch.nn.functional as F

def weighted_bce_loss_with_logits(output, target, weight):
    input = output.view(-1)
    target = target.view(-1)
    weight = weight.view(-1)
    loss = F.binary_cross_entropy_with_logits(input, target)
    weighted_loss = loss * weight
    
    return weighted_loss.mean()
#
