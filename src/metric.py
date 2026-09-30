import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_matrix(gt,pred):
    m=np.zeros((len(gt),len(pred)),np.float32)
    for i,g in enumerate(gt):
        g=g.astype(bool)
        for j,p in enumerate(pred):
            p=p.astype(bool); inter=np.logical_and(g,p).sum(); union=np.logical_or(g,p).sum()
            m[i,j]=inter/union if union else 0
    return m

def pq_score(gt,pred,threshold=0.5):
    m=iou_matrix(gt,pred)
    pairs=[]
    if m.size:
        r,c=linear_sum_assignment(-m)
        pairs=[(i,j,float(m[i,j])) for i,j in zip(r,c) if m[i,j]>threshold]
    tp=len(pairs); fp=len(pred)-tp; fn=len(gt)-tp; s=sum(x[2] for x in pairs)
    den=tp+.5*fp+.5*fn
    return {"pq":s/den if den else 1.0,"sq":s/tp if tp else 0.0,
            "rq":tp/den if den else 1.0,"tp":tp,"fp":fp,"fn":fn,"pairs":pairs}
