import copy
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from models.net_utils import MLP, gen_sineembed_for_position, inverse_sigmoid
from utils.box_utils import generalized_box_iou, box_cxcywh_to_xyxy
import torchvision

from .position_encoding import SeqEmbeddingLearned, SeqEmbeddingSine
from .attention import MultiheadAttention
        

class QueryDecoder(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()    
        d_model = cfg.MODEL.STCAT.HIDDEN
        nhead = cfg.MODEL.STCAT.HEADS
        dim_feedforward = cfg.MODEL.STCAT.FFN_DIM
        dropout = cfg.MODEL.STCAT.DROPOUT
        activation = "relu"
        num_layers = cfg.MODEL.STCAT.DEC_LAYERS
        
        self.d_model = d_model
        self.query_pos_dim = cfg.MODEL.STCAT.QUERY_DIM
        self.nhead = nhead
        self.video_max_len = cfg.INPUT.MAX_VIDEO_LEN
        self.return_weights = cfg.SOLVER.USE_ATTN
        return_intermediate_dec = True
        
        self.template_generator = TemplateGenerator(cfg)
        
        decoder_layer = TransformerDecoderLayer(
            cfg,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation
        )
        
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            return_weights=self.return_weights,
            d_model=d_model,
            query_dim=self.query_pos_dim
        )
        
        temp_decoder_layer = TimeDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation
        )
        
        temp_decoder_norm = nn.LayerNorm(d_model)
        self.temp_decoder = TimeDecoder(
            temp_decoder_layer,
            num_layers,
            temp_decoder_norm,
            return_intermediate=return_intermediate_dec,
            return_weights=True
        )
        
        # The position embedding of global tokens
        if cfg.MODEL.STCAT.USE_LEARN_TIME_EMBED:
            self.time_embed = SeqEmbeddingLearned(self.video_max_len + 1 , d_model)
        else:
            self.time_embed = SeqEmbeddingSine(self.video_max_len + 1, d_model) 
    
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, memory_cache, vis_pos=None, text_cls=None):
        encoded_memory = memory_cache["encoded_memory"]
        memory_mask = memory_cache["mask"]
        durations = memory_cache["durations"]
        fea_map_size = memory_cache["fea_map_size"]   # (H,W) the feature map size
        n_vis_tokens = fea_map_size[0] * fea_map_size[1]
        # the contextual feature to generate dynamic learnable anchors
        frames_cls = memory_cache["frames_cls"]  # n_frames x d_model
        videos_cls = memory_cache["videos_cls"]   # the video-level gloabl contextual token, b x d_model
        
        b = len(durations)
        t = max(durations)
        device = encoded_memory.device
        
        pos_query, temp_query = self.template_generator(
            frames_cls, videos_cls, durations, text_cls
        )
        
        pos_query = pos_query.sigmoid()
        pos_query = torch.split(pos_query, durations, dim=0)
        temp_query = torch.split(temp_query, durations, dim=0)
        tgt = torch.zeros(t, b, self.d_model).to(device) 
        time_tgt = torch.zeros(t, b, self.d_model).to(device) 
        
        # The position embedding of query
        query_pos_embed = torch.zeros(b, t, self.query_pos_dim).to(device)
        query_temporal_embed = torch.zeros(b, t, self.d_model).to(device)
        query_mask = torch.ones(b, t).bool().to(device)
        query_mask[:, 0] = False  # avoid empty masks
        
        for i_dur, dur in enumerate(durations):
            query_mask[i_dur, : dur] = False
            query_pos_embed[i_dur, : dur, :] = pos_query[i_dur]
            query_temporal_embed[i_dur, : dur, :] = temp_query[i_dur]
        
        
        query_pos_embed = query_pos_embed.permute(1, 0, 2)    # [n_frames, bs, 4]
        query_temporal_embed = query_temporal_embed.permute(1,0,2)  # [n_frames, bs, d_model]
        query_time_embed = self.time_embed(t).repeat(1, b, 1) # [n_frames, bs, d_model]
        memory_pos_embed = vis_pos.flatten(2).permute(2, 0, 1)
        memory_pos_embed = torch.cat([memory_pos_embed, torch.zeros_like(encoded_memory[n_vis_tokens:])], dim=0)
        
        outputs = self.decoder(
            tgt,  # t x b x c
            encoded_memory,  # n_tokens x n_frames x c
            tgt_key_padding_mask=query_mask,  # bx(t*n_queries)
            memory_key_padding_mask=memory_mask,  # n_frames * n_tokens
            pos=memory_pos_embed,  # n_tokens x n_frames x c
            query_anchor=query_pos_embed,  # n_queriesx(b*t)xF
            query_time=query_time_embed,
            durations=durations,
            fea_map_size=fea_map_size
        )
        
        outputs_temp = self.temp_decoder(
            time_tgt,
            encoded_memory,
            tgt_key_padding_mask=query_mask,
            memory_key_padding_mask=memory_mask,
            pos=memory_pos_embed,  # n_tokensx(b*t)xF
            query_pos=query_temporal_embed,  # n_queriesx(b*t)xF
            query_time_pos=query_time_embed,
            durations=durations
        )
        
        return outputs, outputs_temp
        

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                    return_weights=False, d_model=256, query_dim=4):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.return_weights = False
        self.query_dim = query_dim
        self.d_model = d_model
        self.attn_map = None
        
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.bbox_embed = None
        
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,   # the pos for feature map
        query_anchor: Optional[Tensor] = None, # the anchor pos embedding
        query_time = None,   # the query time position embedding
        durations=None,
        fea_map_size=None
    ):
        output = tgt
        intermediate = []
        intermediate_weights = []
        ref_anchors = [query_anchor]   # the query pos is like t x b x 4
        
        for layer_id, layer in enumerate(self.layers):
            obj_center = query_anchor[..., :self.query_dim]     # [num_queries, batch_size, 4]
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)  
            query_pos = self.ref_point_head(query_sine_embed)    # generated the position embedding
            
            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)
                
            #################### SMCA with roi feature ############################
            reference_boxes = query_anchor[..., :self.query_dim]
            
            obj_center = query_anchor[..., :self.query_dim]     # [num_queries, batch_size, 4]
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)  
            query_pos = self.ref_point_head(query_sine_embed)
            #########################################################################

            # apply transformation
            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation

            output, temp_weights = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_time_embed=query_time,
                           query_sine_embed=query_sine_embed, durations=durations,
                           is_first=(layer_id == 0), fea_map_size=fea_map_size, reference_boxes=reference_boxes)
            
            # for visualize
            self.attn_map = temp_weights
            
            # iter update
            if self.bbox_embed is not None:
                tmp = self.bbox_embed(output)    # t, b, 4
                tmp[..., :self.query_dim] += inverse_sigmoid(query_anchor) # offset + anchor
                new_query_anchor = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_anchors.append(new_query_anchor)
                    
                query_anchor = new_query_anchor.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.return_weights:
                    intermediate_weights.append(temp_weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            if self.bbox_embed is not None:
                outputs = [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_anchors).transpose(1, 2)
                ]
            else:
                outputs = [
                    torch.stack(intermediate).transpose(1, 2), 
                    query_anchor.unsqueeze(0).transpose(1, 2)
                ]
        
        if self.return_weights:
            return outputs, torch.stack(intermediate_weights)
        else:
            return outputs
        

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        cfg,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_qtime_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_ktime_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        
        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_qtime_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        
        self.from_scratch_cross_attn = cfg.MODEL.STCAT.FROM_SCRATCH
        self.cross_attn_image = None
        self.cross_attn = None
        self.tgt_proj = None
        
        if self.from_scratch_cross_attn:
            self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        else:
            self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model) 
        
        self.nhead = nhead
        self.d_model = d_model
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
        self.point1 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        
        self.smca = True
        if self.smca:
            self.point2 = nn.Sequential(
                nn.Linear(self.d_model // 4 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.nhead * 4),
            )
            nn.init.constant_(self.point2[-1].weight.data, 0)
            nn.init.constant_(self.point2[-1].bias.data, 0)
        else:
            self.point2 = nn.Sequential(
                nn.Linear(self.d_model // 4 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.nhead * 2),
            )
            nn.init.constant_(self.point2[-1].weight.data, 0)
            nn.init.constant_(self.point2[-1].bias.data, 0)
            
        self.attn1 = nn.Linear(self.d_model, self.d_model * self.nhead)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_time_embed=None,
        query_sine_embed = None,
        durations=None,
        is_first = False,
        fea_map_size=None,
        reference_boxes=None,
    ):
        # Apply projections here
        # shape: num_queries x batch_size x 256
        # ========== Begin of Self-Attention =============
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_time = self.sa_qtime_proj(query_time_embed)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_time = self.sa_ktime_proj(query_time_embed)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)
        
        q = q_content + q_time + q_pos
        k = k_content + k_time + k_pos
        
        # Temporal Self attention
        tgt2, weights = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ========== End of Self-Attention =============
        bs = memory.shape[1]
        c = tgt.shape[-1]
        memory_h, memory_w = fea_map_size
        valid_ratio = self.get_valid_ratio(memory_key_padding_mask[:, :memory_h*memory_w].view(bs, memory_h, memory_w))
        memory_2d = memory[:memory_h*memory_w].view(memory_h, memory_w, bs, c)
        memory_2d = memory_2d.permute(2, 3, 0, 1)
        
        reference_boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        reference_boxes_xyxy[:, :, 0] *= memory_w
        reference_boxes_xyxy[:, :, 1] *= memory_h
        reference_boxes_xyxy[:, :, 2] *= memory_w
        reference_boxes_xyxy[:, :, 3] *= memory_h
        reference_boxes_xyxy = reference_boxes_xyxy * valid_ratio.view(bs, 1, 4)
        
        q_content = torchvision.ops.roi_align(
            memory_2d,
            list(torch.unbind(reference_boxes_xyxy, dim=0)),
            output_size=(7, 7),
            spatial_scale=1.0,
            aligned=True)  # (bs * num_queries, c, 7, 7)

        q_content_points = torchvision.ops.roi_align(
            memory_2d,
            list(torch.unbind(reference_boxes_xyxy, dim=0)),
            output_size=(7, 7),
            spatial_scale=1.0,
            aligned=True)  # (bs * num_queries, c, 7, 7)

        q_content_index = q_content_points.view(bs * 1, -1, 7, 7)
        points = self.point1(q_content_index)  # mx: Resemble--> Conv [192, 256, 7, 7]-->[192, 64, 7, 7]
        points = points.reshape(bs * 1, -1)               # [192, 64, 7, 7]-->[192, 3136]
        points = self.point2(points)           # mx: Resemble--> MLP  [192, 3136]-->[192, 16]
        if not self.smca:
            points = points.view(bs * 1, 1, self.nhead, 2).tanh()   # [192, 1, 8, 2]
        else:
            points_scale = points[:, 2 * self.nhead:].reshape(bs, 1, self.nhead, 2).permute(1, 0, 2, 3) # [16, 12, 8, 2]
            points = points[:, :2 * self.nhead].view(bs * 1, 1, self.nhead, 2).tanh()
        
        q_content = F.avg_pool2d(q_content, 7).view(1, bs, -1, c)*self.attn1(tgt).view(1, bs, self.nhead, c).sigmoid()
        
        grid_y, grid_x = torch.meshgrid(torch.arange(0, memory_h), torch.arange(0, memory_w))
        grid = torch.stack((grid_x, grid_y), 2).float().to(tgt.device)
        grid = grid.reshape(-1, 2).unsqueeze(1).repeat(1, bs * 8, 1) # 8 is the heads num.
        
        # SMCA: start
        gau_point = torch.clone(points)
        gau_point[:, :, 0] *= memory_w
        gau_point[:, :, 1] *= memory_h
        gau_point = gau_point.reshape(1, bs, self.nhead, 2)
        gau_point = gau_point * valid_ratio[:, :2].reshape(1, bs, 1, 2)
        gau_point = gau_point.reshape(1, bs * self.nhead, 2)
        gau_distance = (gau_point.unsqueeze(1) - (grid + 0.5).unsqueeze(0)).pow(2)
        gau_scale = points_scale
        gau_scale = gau_scale * gau_scale
        gau_scale = gau_scale.reshape(1, -1, 2).unsqueeze(1)
        gau_distance = (gau_distance * gau_scale).sum(-1)
        gaussian = -(gau_distance - 0).abs() / 8.0         # 8.0 is the number used in SMCA-DETR
        
        tgt = q_content.mean(2).unsqueeze(2).squeeze(0)
        # tgt = q_content.squeeze(0)
        
        # ========== Begin of Cross-Attention =============
        # Time Aligned Cross attention
        t, b, c = tgt.shape    # b is the video number
        n_tokens, bs, f = memory.shape   # bs is the total frames in a batch
        assert f == c   # all the token dim should be same
        
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        
        k_pos = self.ca_kpos_proj(pos)
        
        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content
            
        q = q.view(t, b, self.nhead, c // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(t, b, self.nhead, c // self.nhead)
        
        if self.from_scratch_cross_attn:
            q = torch.cat([q, query_sine_embed], dim=3).view(t, b, c * 2)
        else:
            q = (q + query_sine_embed).view(t, b, c)
            q = q + self.ca_qtime_proj(query_time_embed)
        
        k = k.view(n_tokens, bs, self.nhead, f//self.nhead)
        k_pos = k_pos.view(n_tokens, bs, self.nhead, f//self.nhead)
        
        if self.from_scratch_cross_attn:
            k = torch.cat([k, k_pos], dim=3).view(n_tokens, bs, f * 2)
        else:
            k = (k + k_pos).view(n_tokens, bs, f)
            
        # extract the actual video length query
        clip_start = 0
        device = tgt.device
        if self.from_scratch_cross_attn:
            q_cross = torch.zeros(1,bs,2 * c).to(device)
        else:
            q_cross = torch.zeros(1,bs,c).to(device)
        
        for i_b in range(b):
            q_clip = q[:,i_b,:]   # t x f
            clip_length = durations[i_b]
            q_cross[0,clip_start : clip_start + clip_length] = q_clip[:clip_length]
            clip_start += clip_length
        
        assert clip_start == bs
        
        if self.from_scratch_cross_attn:
            tgt2, attn_weight = self.cross_attn(
                query=q_cross,
                key=k,
                value=v,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                gaussian=gaussian,
            )
        else:
            tgt2, attn_weight = self.cross_attn_image(
                query=q_cross,
                key=k,
                value=v,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                gaussian=gaussian,
            )
     
        # reshape to the batched query
        clip_start = 0
        tgt2_pad = torch.zeros(1,t*b,c).to(device)
        
        for i_b in range(b):
            clip_length = durations[i_b]
            tgt2_pad[0,i_b * t:i_b * t + clip_length] = tgt2[0,clip_start : clip_start + clip_length]
            clip_start += clip_length

        tgt2 = tgt2_pad
        tgt2 = tgt2.view(b, t, f).transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, attn_weight


class TemplateGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.MODEL.STCAT.HIDDEN
        self.pos_query_dim = cfg.MODEL.STCAT.QUERY_DIM
        self.content_proj = nn.Linear(self.d_model, self.d_model)
        self.gamma_proj = nn.Linear(self.d_model, self.d_model)
        self.beta_proj = nn.Linear(self.d_model, self.d_model)
        self.anchor_proj = nn.Linear(self.d_model, self.pos_query_dim)
        
    def forward(
        self, 
        frames_cls=None, 
        videos_cls=None,   # [b, d_model]
        durations=None,   
        text_cls=None     # [b, d_model]
    ):  
        b = len(durations)
        frames_cls_list = torch.split(frames_cls, durations, dim=0)
        content_query = self.content_proj(videos_cls)
        
        pos_query = []
        temp_query = []
        for i_b in range(b):
            frames_cls = frames_cls_list[i_b]
            video_cls = videos_cls[i_b]
            gamma_vec = torch.tanh(self.gamma_proj(video_cls)) 
            beta_vec = torch.tanh(self.beta_proj(video_cls)) 
            pos_query.append(self.anchor_proj(gamma_vec * frames_cls + beta_vec))
            temp_query.append(content_query[i_b].unsqueeze(0).repeat(frames_cls.shape[0],1))
        
        pos_query = torch.cat(pos_query, dim=0)
        temp_query = torch.cat(temp_query, dim=0)
        
        return pos_query, temp_query
        

class TimeDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        return_weights=False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.return_weights = return_weights

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_time_pos: Optional[Tensor] = None,
        durations=None
    ):
        output = tgt

        intermediate = []
        intermediate_weights = []
        # intermediate_cross_weights = []

        for i_layer, layer in enumerate(self.layers):
            output, weights = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                query_time_pos=query_time_pos,
                durations=durations
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.return_weights:
                    intermediate_weights.append(weights)
                    # intermediate_cross_weights.append(cross_weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if not self.return_weights:
                return torch.stack(intermediate).transpose(1, 2)
            else:
                return (
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(intermediate_weights),
                )

        if not self.return_weights:
            return output
        else:
            return output, weights


class TimeDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        no_tsa=False,
    ):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_time_pos: Optional[Tensor] = None,
        durations=None
    ):

        q = k = self.with_pos_embed(tgt, query_pos + query_time_pos)
        # gaussian
        
        # Temporal Self attention
        tgt2, weights = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            # gaussian=gaussian,
        )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        t, b, c = tgt.shape
        n_tokens, bs, f = memory.shape
        
        # extract the actual video length query
        clip_start = 0
        device = tgt.device
        tgt_cross = torch.zeros(1,bs,c).to(device)
        query_pos_cross = torch.zeros(1,bs,c).to(device)
        for i_b in range(b):
            tgt_clip = tgt[:,i_b,:]   # t x f
            query_pos_clip = query_pos[:,i_b,:]
            clip_length = durations[i_b]
            tgt_cross[0,clip_start : clip_start + clip_length] = tgt_clip[:clip_length]
            query_pos_cross[0,clip_start : clip_start + clip_length] = query_pos_clip[:clip_length]
            clip_start += clip_length
        
        assert clip_start == bs
        
        tgt2, _ = self.cross_attn_image(
            query=self.with_pos_embed(tgt_cross, query_pos_cross),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )

        # reshape to the batched query
        clip_start = 0
        tgt2_pad = torch.zeros(1,t*b,c).to(device)
        
        for i_b in range(b):
            clip_length = durations[i_b]
            tgt2_pad[0,i_b * t:i_b * t + clip_length] = tgt2[0,clip_start : clip_start + clip_length]
            clip_start += clip_length

        tgt2 = tgt2_pad
        tgt2 = tgt2.view(b, t, f).transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, weights
        

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
