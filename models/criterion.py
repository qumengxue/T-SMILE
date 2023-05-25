from time import time
import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
import numpy as np

from utils.box_utils import generalized_box_iou, box_cxcywh_to_xyxy
from utils.comm import is_dist_avail_and_initialized, get_world_size
from utils.misc import compute_project_term, NestedTensor, generate_gauss_weight


class VideoSTGLoss(nn.Module):
    """This class computes the loss for VideoSTG Model
    The process happens in two steps:
        1) compute ground truth boxes and the outputs of the model
        2) compute ground truth temporal segment and the outputs sted of model
    """

    def __init__(self, cfg, losses):
        """Create the criterion.
        """
        super().__init__()
        self.cfg = cfg
        self.losses = losses
        self.eos_coef = cfg.SOLVER.EOS_COEF
        
    def loss_vlsimilar(self, outputs, targets, gt_temp_bound, time_mask):
        assert "vl_sim" in outputs
        
        actioness = torch.cat([target["actioness"] for target in targets], dim=0)
        action_idx = np.where(actioness.cpu())[0]
        start_idx, end_idx = action_idx[0], action_idx[-1]
        
        vl_sim = outputs["vl_sim"].sigmoid()
            
        box_masks = torch.cat([target["box_mask"] for target in targets], dim=0).unsqueeze(1)
        
        dsample_box_mask = F.interpolate(box_masks, vl_sim.shape[-2:], mode="bilinear")
        
        all_box_mask = torch.zeros_like(vl_sim.unsqueeze(1))
        all_box_mask[start_idx: end_idx+1] = dsample_box_mask
        
        weight = torch.full(actioness.shape, self.eos_coef, device=actioness.device)
        temp_bound = gt_temp_bound
        weight[temp_bound[0][0] : temp_bound[0][1] + 1] = 1
        loss_vl_sim = (compute_project_term(vl_sim.unsqueeze(1), all_box_mask)*weight).mean()
        
        losses = {"loss_vlsimilar": loss_vl_sim}
        
        return losses

    def loss_boxes(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        
        target_actioness = torch.stack([target["actioness"] for target in targets], dim=0).float()
        gauss_center = torch.tensor(int((torch.where(target_actioness[0]==1)[0][0]+torch.where(target_actioness[0]==1)[0][-1])/2)).unsqueeze(0).to(target_actioness.device)
        sigma = target_actioness[0].sum()/2
        eps = 1e-6
        target_gauss = (
            -(
                (
                    torch.arange(len(target_actioness[0]))[None, :].to(target_actioness.device)
                    - gauss_center
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
            
        src_boxes = outputs["pred_boxes"]
        target_boxes = torch.cat([target["boxs"].bbox for target in targets], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        weight = F.normalize(target_gauss + eps, p=1, dim=1).permute(1,0)
        
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / max(torch.nonzero(targets[0]['actioness']).shape[0], 1)

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        
        losses["loss_giou"] = loss_giou.sum() / max(torch.nonzero(targets[0]['actioness']).shape[0], 1)

        return losses
    
    def loss_actioness(self, outputs, targets, gt_temp_bound, time_mask=None):
        assert "pred_actioness" in outputs
        losses = {}
        pred_actioness = outputs['pred_actioness'].squeeze(-1)
        target_actioness = torch.stack([target["actioness"] for target in targets], dim=0).float()
        weight = torch.full(pred_actioness.shape, self.eos_coef, device=pred_actioness.device)
        sigma = self.cfg.SOLVER.SIGMA
        eps = 1e-6
        ############ generate target gaussian ############
        target_gauss = torch.full(pred_actioness.shape, 0., device=pred_actioness.device)
        
        for i_b in range(len(weight)):
            temp_bound = gt_temp_bound[i_b]
            weight[i_b][temp_bound[0] : temp_bound[1] + 1] = 1
            gauss_center = torch.tensor(int((torch.where(target_actioness[i_b]==1)[0][0]+torch.where(target_actioness[i_b]==1)[0][-1])/2)).unsqueeze(0).to(target_actioness.device)
            sigma = target_actioness[i_b].sum()/2
            target_gauss[i_b] = (
                -(
                    (
                        torch.arange(len(target_actioness[i_b]))[None, :].to(target_actioness.device)
                        - gauss_center
                    )
                    ** 2
                )
                / (2 * sigma ** 2)
            ).exp()  # gaussian target
            
        loss_actioness = F.binary_cross_entropy_with_logits(pred_actioness, \
                target_actioness, weight=weight, reduction='none')
        
        loss_actioness = loss_actioness * time_mask
        losses["loss_actioness"] = loss_actioness.mean()
        return losses

    def loss_sted(self, outputs, num_boxes, targets, gt_temp_bound, positive_map, time_mask=None):
        assert "pred_sted" in outputs
        sted = outputs["pred_sted"]
        losses = {}
        
        target_action = torch.stack([target["ori_actioness"] for target in targets], dim=0).float()
        gt_frame = torch.stack([torch.nonzero(target)[0] for target in target_action], dim=0)
        
        target_start = torch.tensor([x[0] for x in gt_temp_bound], dtype=torch.long).to(sted.device)
        target_end = torch.tensor([x[1] for x in gt_temp_bound], dtype=torch.long).to(sted.device)
        sted = sted.masked_fill(~time_mask[:, :, None], -1e32)  # put very low probability on the padded positions before softmax
        eps = 1e-6
        
        sigma = self.cfg.SOLVER.SIGMA
        # sigma = 4 * sigma
        start_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_start[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target

        #################################################
        pred_start_prob_wo_sigmoid = sted[:, :, 0]
        pred_start_prob = (sted[:, :, 0]).sigmoid()
        start_distrib[..., gt_frame:] = 0.
        start_weight = torch.full(pred_start_prob.shape, self.eos_coef, device=pred_start_prob.device)
        start_weight[..., gt_frame:] = 1.
        # loss_start = F.binary_cross_entropy_with_logits(pred_start_prob, start_target, weight=start_weight, reduction='none')
        
        ################ mil loss ########################
        start_weight = (start_distrib>0.5).to(torch.float32)
        pos_num = torch.where(start_weight[0]==1)[0].shape[0]
        pred_pos, indice_pos = torch.topk(pred_start_prob[0]*start_weight[0], k=round(pos_num*0.51))
        new_start_weight = torch.full(pred_start_prob.shape, self.eos_coef, device=pred_start_prob.device)
        start_target = torch.full(pred_start_prob.shape, 0., device=pred_start_prob.device)
        new_start_weight[0, indice_pos] = 1.
        start_target[0, indice_pos] = 1.
        loss_start = F.binary_cross_entropy_with_logits(pred_start_prob_wo_sigmoid, start_target, weight=new_start_weight, reduction='none')
        #################################################
        
        loss_start = loss_start * time_mask
        end_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_end[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        
        ################################################
        pred_end_prob_wo_sigmoid = sted[:, :, 1]
        pred_end_prob = (sted[:, :, 1]).sigmoid()
        end_distrib[..., :gt_frame] = 0.
        end_weight = torch.full(pred_end_prob.shape, self.eos_coef, device=pred_start_prob.device)
        end_weight[..., gt_frame:] = 1.
        
        ################ mil loss ########################
        end_weight = (end_distrib>0.5).to(torch.float32)
        pos_num = torch.where(end_weight[0]==1)[0].shape[0]
        # pred_end_prob = F.avg_pool1d(pred_end_prob, kernel_size=3, stride=1, padding=1)
        pred_pos, indice_pos = torch.topk(pred_end_prob[0]*end_weight[0], k=round(pos_num*0.51))
        new_end_weight = torch.full(pred_end_prob.shape, self.eos_coef, device=pred_end_prob.device)
        end_target = torch.full(pred_end_prob.shape, 0., device=pred_end_prob.device)
        new_end_weight[0, indice_pos] = 1.
        end_target[0, indice_pos] = 1.
        loss_end = F.binary_cross_entropy_with_logits(pred_end_prob_wo_sigmoid, end_target, weight=new_end_weight, reduction='none')
        ################################################
        
        loss_end = loss_end * time_mask
        loss_sted = loss_start + loss_end
        losses["loss_sted"] = loss_sted.mean()
        
        return losses
    
    def loss_guided_attn(
        self, outputs, num_boxes, gt_temp_bound, positive_map, time_mask=None
    ):
        """Compute guided attention loss
        targets dicts must contain the key "weights" containing a tensor of attention matrices of dim [B, T, T]
        """
        weights = outputs["weights"]  # BxTxT
        
        positive_map = positive_map + (~time_mask)  # the padded positions also have to be taken out
        eps = 1e-6  # avoid log(0) and division by 0

        loss = -(1 - weights + eps).log()
        loss = loss.masked_fill(positive_map[:, :, None], 0)
        nb_neg = (~positive_map).sum(1) + eps
        loss = loss.sum(2) / nb_neg[:, None]  # sum on the column
        loss = loss.sum(1)  # mean on the line normalized by the number of negatives
        loss = loss.mean()  # mean on the batch
        
        losses = {"loss_guided_attn": loss}
        return losses

    def get_loss(
        self, loss, outputs, targets, num_boxes, gt_temp_bound, positive_map, time_mask, **kwargs,
    ):
        loss_map = {
            "boxes": self.loss_boxes,
            "sted": self.loss_sted,
            "guided_attn": self.loss_guided_attn,
            "actioness": self.loss_actioness,
            "loss_vlsimilar": self.loss_vlsimilar,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss in ["sted", "guided_attn"]:
            return loss_map[loss](
                outputs, num_boxes, targets, gt_temp_bound, positive_map, time_mask, **kwargs
            )
        if loss in ["actioness", "loss_vlsimilar"]:
            return loss_map[loss](outputs, targets, gt_temp_bound, time_mask, **kwargs)
        
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    def forward(self, outputs, targets, durations):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        max_duration = max(durations)
        device = outputs["pred_boxes"].device
        gt_bbox_slice, gt_temp_bound = [], []
        
        for i_dur, (duration, target) in enumerate(zip(durations, targets)):
            # inter = torch.where(torch.ones_like(target['actioness']))[0].cpu().numpy().tolist()
            inter = torch.where(target['actioness'])[0].cpu().numpy().tolist()
            # inter = torch.where(target['ori_actioness'])[0].cpu().numpy().tolist()
            gt_temp_bound.append([inter[0],inter[-1]])
            gt_bbox_slice.extend(list(range(i_dur * max_duration + inter[0], i_dur * max_duration + inter[-1] + 1)))
            
        gt_bbox_slice = torch.LongTensor(gt_bbox_slice).to(device)
        outputs["pred_boxes"] = outputs["pred_boxes"][gt_bbox_slice]

        for i_aux in range(len(outputs["aux_outputs"])):
            outputs["aux_outputs"][i_aux]["pred_boxes"] = outputs["aux_outputs"][i_aux]["pred_boxes"][gt_bbox_slice]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(target['boxs']) for target in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # computer the temporal mask, used for guided-attn
        b = len(durations)
        time_mask = torch.zeros(b, max(durations)).bool().to(device)
        for i_dur, duration in enumerate(durations):
            time_mask[i_dur, :duration] = True
    
        positive_map = torch.zeros(time_mask.shape, dtype=torch.bool)
        for k, idx in enumerate(gt_temp_bound):
            if idx[0] < 0:  # empty intersection
                continue
            positive_map[k][idx[0] : idx[1] + 1].fill_(True)

        positive_map = positive_map.to(time_mask.device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_boxes, gt_temp_bound, positive_map, time_mask))
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    if loss == "loss_vlsimilar":
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_boxes, gt_temp_bound, positive_map, time_mask, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses