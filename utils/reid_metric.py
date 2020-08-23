# encoding: utf-8


import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func, arm_eval_func
from .re_ranking import re_ranking


class R1_mAP_arm(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_arm, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.g_f_feats = []
        self.part_feats = []
        self.part_visibles = []
        self.pids = []
        self.camids = []

    def update(self, output):
        g_f_feat, part_feat, part_visible, pid, camid = output
        self.g_f_feats.append(g_f_feat)
        self.part_feats.append(part_feat)
        self.part_visibles.extend(np.asarray(part_visible))
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        g_f_feats = torch.cat(self.g_f_feats, dim=0)
        part_feats = torch.cat(self.part_feats, dim=0)
        # query
        q_gf_f = g_f_feats[:self.num_query]
        q_part_f = part_feats[:self.num_query]
        q_pids = torch.Tensor(self.pids[:self.num_query])
        q_camids = torch.Tensor(self.camids[:self.num_query])
        q_part_visible = torch.Tensor(self.part_visibles[:self.num_query])
        
        # gallery
        g_gf_f = g_f_feats[self.num_query:]
        g_part_f = part_feats[self.num_query:]
        g_pids = torch.Tensor(self.pids[self.num_query:])
        g_camids = torch.Tensor(self.camids[self.num_query:])
        g_part_visible = torch.Tensor(self.part_visibles[self.num_query:])
        # tgpl:total gallery part label; tgf:total gallery partial feature; tgf2:total gallery pose-guided global feature; tgl:total gallery label; tgc: total gallery camera id
        # tqpl:total query part label; tqf:total query partial feature; tqf2:total query pose-guided global feature; tql:total query label; tqc: total query camera id
        count=0
        CMC=torch.IntTensor(len(g_pids)).zero_()    
        ap=0.0
      
        for qf,qf2,qpl,ql,qc in zip(q_part_f,q_gf_f,q_part_visible,q_pids,q_camids):
            (ap_tmp, CMC_tmp),index = arm_eval_func(qf,qf2,qpl,ql,qc,g_part_f,g_gf_f,g_part_visible,g_pids,g_camids) #
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            count+=1
        CMC = CMC.float()
        CMC = CMC/count #average CMC
        mAP = ap/count

        return CMC, mAP

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        print('no_arm evaluate')
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP