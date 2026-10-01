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
