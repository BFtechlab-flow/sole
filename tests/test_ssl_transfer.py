import torch
import pytest

from src.model import FILANet
from src.ssl_pretrain import MaskedSolarAutoencoder, load_ssl_encoder


def test_ssl_encoder_transfers_strictly_into_filanet():
    source = MaskedSolarAutoencoder("resnet18", pretrained=False)
    key = next(iter(source.encoder.state_dict()))
    with torch.no_grad():
        state = source.encoder.state_dict()
        state[key].fill_(0.123)
        source.encoder.load_state_dict(state)

    checkpoint = {
        "encoder": source.encoder.state_dict(),
        "encoder_name": "resnet18",
        "input_channels": 4,
    }
    target = FILANet("resnet18", pretrained=False, fpn_channels=32, in_chans=4, advanced={"enabled": False})
    load_ssl_encoder(target, checkpoint, strict=True)
    assert torch.allclose(target.encoder.state_dict()[key], source.encoder.state_dict()[key])


def test_ssl_encoder_rejects_wrong_channel_count():
    target = FILANet("resnet18", pretrained=False, fpn_channels=32, in_chans=4, advanced={"enabled": False})
    with pytest.raises(ValueError):
        load_ssl_encoder(target, {"input_channels": 3, "encoder": target.encoder.state_dict()})
