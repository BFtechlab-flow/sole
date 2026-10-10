import numpy as np
import torch
from src.inference import predict_tiled

class EchoModel(torch.nn.Module):
    def forward(self,x):
        return {
            "region":x[:,0:1],
            "centerline":x[:,1:2],
            "boundary":x[:,2:3],
            "distance":x[:,3:4],
        }

def test_tta_inverse_transforms_preserve_scalar_prediction_geometry():
    rng=np.random.default_rng(123)
    x=rng.normal(size=(4,31,29)).astype(np.float32)
    pred=predict_tiled(EchoModel().eval(),x,"cpu",tile=16,overlap=4,tta=True)
    expected=1/(1+np.exp(-x))
    for i,key in enumerate(("region","centerline","boundary","distance")):
        assert np.allclose(pred[key],expected[i],atol=1e-6)


class EmbeddingModel(torch.nn.Module):
    """Region from the input; an instance embedding whose coordinates depend on
    the view (flip) and on the model (``offset``), as nothing constrains them."""

    def __init__(self, offset):
        super().__init__()
        self.offset = float(offset)

    def forward(self, x):
        b, _, h, w = x.shape
        ramp = torch.linspace(0, 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
        emb = torch.cat([ramp + self.offset, 1 - ramp, torch.full_like(ramp, self.offset)], dim=1)
        return {"region": x[:, 0:1], "centerline": x[:, 1:2], "boundary": x[:, 2:3],
                "distance": x[:, 3:4], "instance_embedding": emb}


def test_embedding_comes_from_identity_view_and_first_model_only():
    from src.inference import ensemble_predict

    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 24, 20)).astype(np.float32)
    m1, m2 = EmbeddingModel(0.0).eval(), EmbeddingModel(5.0).eval()
    single = predict_tiled(m1, x, "cpu", tile=24, overlap=0, tta=True)
    plain = predict_tiled(m1, x, "cpu", tile=24, overlap=0, tta=False)
    # TTA must not average the embedding over flipped views
    assert np.allclose(single["instance_embedding"], plain["instance_embedding"], atol=1e-6)
    both = ensemble_predict([m1, m2], x, "cpu", tile=24, overlap=0, tta=True)
    # the ensemble keeps model 1's embedding untouched ...
    assert np.allclose(both["instance_embedding"], single["instance_embedding"], atol=1e-6)
    # ... while probability maps are averaged over the models
    assert np.allclose(both["region"], single["region"], atol=1e-6)
