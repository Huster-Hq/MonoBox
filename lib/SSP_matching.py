import torch
from torch import nn
import torch.nn.functional as F
import pdb

class SSP_MatchingNet(nn.Module):
    def __init__(self, refine=False):
        super(SSP_MatchingNet, self).__init__()
        self.refine = refine

    def forward(self, feature, mask_q, kernel=False, FP=None, BP=None):
        h, w = mask_q.shape[-2:]
        feature_q = feature
        mask = mask_q
        if kernel:
            SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, mask)
            FP_1 = SSFP_1 * 1.
            BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
        else:
            FP_1 = FP
            BP_1 = BP
        out_1 = self.similarity_func(feature_q, FP_1, BP_1)

        ##################### SSP Refinement #####################
        if self.refine:
            SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feature_q, out_1)
            FP_2 = SSFP_2 * 1
            BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7
            FP_2 = FP_1 * 0.4 + FP_2 * 0.6
            BP_2 = BP_1 * 0.4 + BP_2 * 0.6
            out_2 = self.similarity_func(feature_q, FP_2, BP_2)
            out_2 = out_2 * 0.7 + out_1 * 0.3
        
        out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)
        if self.refine:
            out_2 = F.interpolate(out_2, size=(h, w), mode="bilinear", align_corners=True)
            out_ls = [out_2, out_1]
        else:
            out_ls = [out_1]
             
        if kernel:
            return FP_1, BP_1
        else:
            return out_ls[0]

    def SSP_func(self, feature_q, out):
        bs = feature_q.shape[0]
        pred_1 = out.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.5 #0.9 #0.6
            bg_thres = 0.8 #0.6
            cur_feat = feature_q[epi].contiguous().view(feature_q.shape[1], -1)   #[1024,3600]
            f_h, f_w = feature_q[epi].shape[-2:]    #60,60
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)
            else:
                a = (pred_fg[epi] > fg_thres).sum()
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices] #.mean(-1)  #[1024,12]
            if (pred_bg[epi] > bg_thres).sum() > 0:
                b = (pred_bg[epi] > bg_thres).sum()
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)  #[1024,2365]
            else:
                b = (pred_bg[epi] > bg_thres).sum()
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices] #.mean(-1)
            
            # global proto
            fg_proto = fg_feat.mean(-1)   #[1024]
            bg_proto = bg_feat.mean(-1)   #[1024]
            fg_ls.append(fg_proto.unsqueeze(0))  #[[1,1024]]
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / (torch.norm(fg_feat, 2, 0, True))#+1e-6) # 1024, N1
            bg_feat_norm = bg_feat / (torch.norm(bg_feat, 2, 0, True))#+1e-6) # 1024, N2
            cur_feat_norm = cur_feat / (torch.norm(cur_feat, 2, 0, True))#+1e-6) # 1024, N3
            if torch.isnan(bg_feat_norm).any():  # is nan
                print('bg_feature_norm: nan')
            if torch.isnan(fg_feat_norm).any():  # is not nan
                print('bg_feat_norm: nan')

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1    #相似矩阵
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2    #相似矩阵

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(feature_q.shape[1], f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(feature_q.shape[1], f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        if torch.isnan(feature_q).any():
            print('feature_q: nan')
        if torch.isnan(fg_proto).any():
            print('fg_proto: nan')
        if torch.isnan(bg_proto).any():
            print('bg_proto: nan')
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        if torch.isnan(similarity_fg).any():
            print('similarity_fg: nan')
        if torch.isnan(similarity_bg).any():
            print('similarity_bg: nan')

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) #* 10.0

        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature
