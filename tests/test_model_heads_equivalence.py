"""Heads at stride 4 + upsampling == upsampling + heads (memory-saving forward)."""
import torch
import torch.nn.functional as F

from src.model import FILANet


def test_low_resolution_heads_match_full_resolution_heads():
    torch.manual_seed(0)
    model = FILANet("resnet18", False, 16, 4, advanced={"enabled": True, "instance_embedding_head": True}).eval()
    x = torch.randn(2, 4, 96, 128)
    captured = {}
    model.limb_gate.register_forward_hook(lambda m, i, o: captured.__setitem__("z", o))
    with torch.no_grad():
        out = model(x)
        z_up = F.interpolate(captured["z"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        ref = model._heads(z_up)
    assert set(out) == set(ref)
    for k in ref:
        assert out[k].shape == ref[k].shape
        assert torch.allclose(out[k], ref[k], atol=1e-5), k
