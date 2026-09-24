"""Project losses; optional 2D soft-clDice following Shit et al., CVPR 2021."""
import common
import torch
from torch.nn import functional as F
from octa.losses import BCEDiceLoss


def erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)),
                         -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def skeleton(x, iterations=10):
    # Soft morphological skeleton; sum per sample, not across the batch.
    result = torch.zeros_like(x)
    for _ in range(iterations + 1):
        smaller = erode(x)
        opened = F.max_pool2d(smaller, 3, 1, 1)
        delta = F.relu(x - opened)
        result = result + F.relu(delta - result * delta)
        x = smaller
    return result


def soft_cldice(logits, target, iterations=10):
    p, y = logits.float().sigmoid(), target.float()
    sp, sy = skeleton(p, iterations), skeleton(y, iterations)
    dims = (1, 2, 3)
    precision = ((sp * y).sum(dims) + 1) / (sp.sum(dims) + 1)
    recall = ((sy * p).sum(dims) + 1) / (sy.sum(dims) + 1)
    return (1 - 2 * precision * recall / (precision + recall).clamp_min(1e-8)).mean()


def retention_loss(logits, anchor, target):
    """Only discourage loss of true positive probability; no negative pseudo-labels."""
    p0 = anchor.detach().float().sigmoid()
    keep = target.float() * (p0 >= 0.5)
    penalty = F.relu(p0 - logits.float().sigmoid()) * keep
    return penalty.sum() / keep.sum().clamp_min(1)


def objective(output, target, config):
    logits = output["logits"].float()
    criterion = BCEDiceLoss()
    seg = criterion(logits, target.float())
    aux = torch.zeros_like(seg)
    retain = torch.zeros_like(seg)
    if config["model"] == "ugr":
        aux = criterion(output["anchor"].float(), target.float())
        retain = retention_loss(logits, output["anchor"], target)
    topology = soft_cldice(logits, target) if config["cldice_weight"] > 0 else torch.zeros_like(seg)
    total = seg + config["aux_weight"] * aux + config["retention_weight"] * retain + config["cldice_weight"] * topology
    return total, {"seg_loss": seg.detach(), "aux_loss": aux.detach(),
                   "retention_loss": retain.detach(), "topology_loss": topology.detach()}
