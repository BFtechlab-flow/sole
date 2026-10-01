import numpy as np
from src.data import apply_geometric_transform

def test_geometric_transform_keeps_image_and_targets_aligned():
    base=np.arange(20,dtype=np.float32).reshape(4,5)
    image=np.stack([base,base+100],axis=0)
    targets={
        "region":base.copy(),
        "centerline":base.copy()+10,
        "boundary":base.copy()+20,
        "distance":base.copy()+30,
    }
    x,out=apply_geometric_transform(image,targets,hflip=True,vflip=True)
    expected=np.flipud(np.fliplr(base))
    assert np.array_equal(x[0],expected)
    assert np.array_equal(out["region"],expected)
    assert np.array_equal(out["centerline"],expected+10)
    assert np.array_equal(out["boundary"],expected+20)
    assert np.array_equal(out["distance"],expected+30)
