def weighted_bce_loss_with_logits(output, target, weight):
    input = output.view(-1)
    target = target.view(-1)
    weight = weight.view(-1)
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    weighted_loss = loss * weight
    
    return weighted_loss.mean()
#
