import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.morphology import remove_small_objects,skeletonize,binary_closing,disk
from skimage.segmentation import watershed

def reconstruct_instances(region,center,boundary,distance,region_threshold=.45,center_threshold=.35,
                          min_region_area=64,min_instance_area=250,**kwargs):
    fg=remove_small_objects(region>region_threshold,min_size=min_region_area)
    seeds=skeletonize(binary_closing((center>center_threshold)&fg,footprint=disk(1)))
    markers=label(seeds)
    if markers.max()==0 and fg.any():
        d=distance_transform_edt(fg); markers=label(d>np.percentile(d[fg],80))
    inst=watershed((-distance+.9*boundary).astype(np.float32),markers=markers,mask=fg)
    return [(inst==i).astype(np.uint8) for i in range(1,inst.max()+1) if (inst==i).sum()>=min_instance_area]
