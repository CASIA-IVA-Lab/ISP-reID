# encoding: utf-8

import numpy as np
import scipy.io
import torch
import time
import os
import torch.nn.functional as F


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        #print(q_idx,AP,indices[q_idx][keep][0:30])

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def arm_eval_func(qf,qf2,qpl,ql,qc,gf,gf2,gpl,gl,gc):
    qf=qf.cuda()
    gf=gf.cuda()
    gpl=gpl.cuda()
    qpl=qpl.cuda()
    qf2=qf2.cuda()
    gf2=gf2.cuda()
    #######Calculate the distance of pose-guided global features

    query2 = qf2
    
    
    qf2=qf2.expand_as(gf2)

    q2=F.normalize(qf2,p=2,dim=1)
    g2=F.normalize(gf2,p=2,dim=1)
    s2=q2*g2
    s2=s2.sum(1) #calculate the cosine distance 
    s2=(s2+1.)/2 # convert cosine distance range from [-1,1] to [0,1], because occluded part distance is set to 0

    ########Calculate the distance of partial features
    query = qf
    overlap=gpl*qpl
    overlap=overlap.view(-1,gpl.size(1)) #Calculate the shared region part label
    
    qf=qf.expand_as(gf)

    q=F.normalize(qf,p=2,dim=2)
    g=F.normalize(gf,p=2,dim=2)
    s=q*g
    
    s=s.sum(2) #Calculate the consine distance 
    s=(s+1.)/2 # convert cosine distance range from [-1,1] to [0,1]
    s=s*overlap
    s=(s.sum(1)+s2)/(overlap.sum(1)+1)
    s=s.data.cpu()
    ###############
    ###############
    score=s.numpy()
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp,index


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc
