import warnings

import numpy as np
from pycocotools import mask as mask_utils


def encode_counts(mask):
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    c = r["counts"]
    return c.decode("utf-8") if isinstance(c, bytes) else c


def decode_counts(counts, h=2048, w=2048):
    c = counts.encode("utf-8") if isinstance(counts, str) else counts
    # pycocotools currently emits a NumPy 2.x DeprecationWarning internally.
    # Keep suppression local so genuine warnings elsewhere remain visible.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        decoded = mask_utils.decode({"size": [h, w], "counts": c})
    return decoded.astype(np.uint8)


def assert_roundtrip(mask):
    c = encode_counts(mask)
    if not np.array_equal(mask.astype(np.uint8), decode_counts(c, *mask.shape)):
        raise AssertionError("RLE round-trip mismatch")
    return c
