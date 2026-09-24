import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model import UGRFusion
from losses import retention_loss, soft_cldice

torch.set_num_threads(1)


def test_identity_at_initialization_and_odd_shapes():
    model = UGRFusion(base=4, oct_base=2).eval()
    for shape in ((32, 32), (47, 49)):
        x = torch.rand(2, 2, *shape)
        with torch.no_grad():
            out = model.forward_details(x)
            anchor = model.anchor(x[:, :1])
        assert out["logits"].shape == (2, 1, *shape)
        torch.testing.assert_close(out["logits"], anchor)
        assert out["correction"].count_nonzero() == 0


def test_oct_gradients_after_zero_head_learns_and_correction_bound():
    model = UGRFusion(base=4, oct_base=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for step in range(2):
        optimizer.zero_grad()
        x = torch.rand(2, 2, 32, 32, requires_grad=True)
        out = model.forward_details(x)
        torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], torch.ones_like(out["logits"])).backward()
        if step == 1:
            assert torch.isfinite(x.grad).all()
            assert x.grad[:, 1].abs().sum() > 0
        optimizer.step()
    for refine in model.refiners:
        torch.nn.init.constant_(refine.residual.bias, 100)
    out = model.forward_details(x.detach())
    assert out["correction"].abs().max() <= 2.00001


def test_capacity_control_ignores_oct():
    model = UGRFusion(base=4, oct_base=2, aux_source="octa").eval()
    for refine in model.refiners:
        torch.nn.init.normal_(refine.residual.weight)
    x = torch.rand(2, 2, 32, 32)
    changed = x.clone()
    changed[:, 1] = torch.rand_like(changed[:, 1])
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(changed))


def test_retention_only_penalizes_true_foreground_suppression():
    anchor = torch.full((1, 1, 4, 4), 2.)
    logits = torch.zeros_like(anchor, requires_grad=True)
    assert retention_loss(logits, anchor, torch.zeros_like(anchor)) == 0
    penalty = retention_loss(logits, anchor, torch.ones_like(anchor))
    assert penalty > 0
    penalty.backward()
    assert (logits.grad < 0).all()


def test_topology_finite_for_empty_and_vessel_masks():
    for full in (False, True):
        y = torch.zeros(2, 1, 32, 32)
        if full:
            y[:, :, 4:28, 15:17] = 1
        z = torch.randn_like(y, requires_grad=True)
        loss = soft_cldice(z, y, iterations=3)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


def test_shuffle_has_no_fixed_points():
    from evaluate import Perturbed
    dataset = [{"image": torch.stack([torch.zeros(8, 8), torch.full((8, 8), float(i))])} for i in range(10)]
    modified = Perturbed(dataset, "shuffle", 42)
    for i in range(10):
        assert modified.mapping[i] != i
        assert modified[i]["image"][1, 0, 0] != i
        assert dataset[i]["image"][1, 0, 0] == i
