# encoding: utf-8

import logging

import torch
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_arm


def create_supervised_evaluator(model, metrics,
                                device=None, with_arm=False):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.cuda()
            if with_arm:
                g_f_feat, part_feat, part_visible, _ = model(data)
                return g_f_feat, part_feat, part_visible, pids, camids
            else:
                feat, _ = model(data)
                return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE
    with_arm = cfg.TEST.WITH_ARM

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        if with_arm:
            evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_arm(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device, with_arm=with_arm)
        else:
            evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device, with_arm=with_arm)

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
