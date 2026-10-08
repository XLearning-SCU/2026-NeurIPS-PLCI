import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
L2norm = nn.functional.normalize

def sinkhorn_log_domain(p, q, C, reg=0.05, niter=50, thresh=1e-6):
    def M(u, v):
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / reg

    u = torch.zeros_like(p)
    v = torch.zeros_like(q)

    for i in range(niter):
        u1 = u.clone()
        u = reg * (torch.log(p) - torch.logsumexp(M(u, v), dim=1, keepdim=True).squeeze()) + u
        v = reg * (torch.log(q) - torch.logsumexp(M(u, v).mT, dim=1, keepdim=True).squeeze()) + v
        
        err = torch.norm(u - u1)
        if err < thresh:
            break
            
    pi = torch.exp(M(u, v)).float()
    return pi


class Ours(torch.nn.Module):
    def __init__(
        self,
        n_views,
        layer_dims,
        temperature,
        n_classes,
        drop_rate=0.5,
        missing_loss_weight=1.0,
        sparse_topk=3,
        sparse_temp=0.25,
        missing_start_epoch=20,
        min_missing_conf=0.2,
        loss_distance=0.8,
    ):
        super(Ours, self).__init__()
        if n_views != 2:
            raise NotImplementedError('The current joint training implementation assumes exactly 2 views.')

        self.n_views = n_views
        self.n_classes = n_classes
        self.missing_loss_weight = missing_loss_weight
        self.sparse_topk = sparse_topk
        self.sparse_temp = sparse_temp
        self.missing_start_epoch = missing_start_epoch
        self.min_missing_conf = min_missing_conf
        self.loss_distance = loss_distance

        self.online_encoder = nn.ModuleList([FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)])
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.cross_view_decoder = nn.ModuleList([MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])

        self.cl = ContrastiveLoss(temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]

        self.register_buffer('prototypes_view0', torch.zeros(n_classes, layer_dims[0][-1]))
        self.register_buffer('prototypes_view1', torch.zeros(n_classes, layer_dims[1][-1]))
        self.register_buffer('R_0to1', torch.eye(n_classes))
        self.register_buffer('R_1to0', torch.eye(n_classes))
        self.prototypes_initialized = False
        self.proto_tau = 0.05 + 0.02 * math.log(n_classes)
        self.latest_debug = {}

    def forward(self, data, mask, momentum, warm_up, idx=None, epoch=None, singular_thresh=None):
        self._update_target_branch(momentum)
        device = data[0].device

        z = self._encode_masked(self.online_encoder, data, mask)
        with torch.no_grad():
            z_t = self._encode_masked(self.target_encoder, data, mask)
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]

        present_idx = [mask[:, i] for i in range(self.n_views)]
        complete_mask = (mask.sum(dim=1) == self.n_views)
        complete_idx = torch.where(complete_mask)[0]
        debug_info = {
            'keep_ratio': None,
            'eta_mean': None,
            'alpha_mean': None,
            'eta_max': None,
            'peak_mean': None,
            'entropy_mean': None,
            'missing_feat': None,
            'missing_assign': None,
        }

        # ------------------------------------------------------------------
        # 1) cache per-sample inter-view loss for complete samples only
        # ------------------------------------------------------------------
        if complete_idx.numel() > 1:
            with torch.no_grad():
                eye_complete = torch.eye(complete_idx.numel(), device=device)
                l_inter_0to1_ps = self.cl(p[0][complete_idx], z_t[1][complete_idx], eye_complete, return_per_sample=True)
                l_inter_1to0_ps = self.cl(p[1][complete_idx], z_t[0][complete_idx], eye_complete, return_per_sample=True)
                per_sample_loss_raw = (l_inter_0to1_ps + l_inter_1to0_ps) / 2
                if idx is not None:
                    idx = idx.to(device)
                    max_idx = idx.max().item()
                    if (not hasattr(self, 'inter_loss_global')) or max_idx >= self.inter_loss_global.shape[0]:
                        new_size = max(max_idx + 1, 50000)
                        self.register_buffer('inter_loss_global', torch.full((new_size,), 10.0, device=device))
                    self.inter_loss_global[idx[complete_idx]] = per_sample_loss_raw.detach()

        # ------------------------------------------------------------------
        # 2) build intra-view affinity only on present samples
        # ------------------------------------------------------------------
        intra_mp = []
        for i in range(self.n_views):
            if present_idx[i].sum() <= 1:
                intra_mp.append(None)
            elif warm_up:
                intra_mp.append(torch.eye(int(present_idx[i].sum().item()), device=device))
            else:
                intra_mp.append(self.kernel_affinity(z_t[i][present_idx[i]]))

        # ------------------------------------------------------------------
        # 3) complete-sample branch: preserve inter-view learning
        # ------------------------------------------------------------------
        l_inter_complete = torch.tensor(0.0, device=device)
        if complete_idx.numel() > 1:
            if warm_up:
                if epoch is not None and epoch < 10:
                    l_inter_complete = torch.tensor(0.0, device=device)
                else:
                    eye_complete = torch.eye(complete_idx.numel(), device=device)
                    l_inter_0_ps = self.cl(p[0][complete_idx], z_t[1][complete_idx], eye_complete, return_per_sample=True)
                    l_inter_1_ps = self.cl(p[1][complete_idx], z_t[0][complete_idx], eye_complete, return_per_sample=True)
                    curr_loss = (l_inter_0_ps + l_inter_1_ps) / 2

                    with torch.no_grad():
                        L = curr_loss.detach().view(-1, 1)
                        c1, c2 = L.min(), L.max()
                        for _ in range(5):
                            d1, d2 = torch.abs(L - c1), torch.abs(L - c2)
                            assign = (d1 < d2).float()
                            sum_assign = assign.sum()
                            if sum_assign > 0:
                                c1 = (L * assign).sum() / sum_assign
                            sum_unassign = (1 - assign).sum()
                            if sum_unassign > 0:
                                c2 = (L * (1 - assign)).sum() / sum_unassign
                        clean_center, noisy_center = torch.min(c1, c2), torch.max(c1, c2)
                        if noisy_center - clean_center < self.loss_distance:
                            robust_mask = torch.ones_like(curr_loss)
                        else:
                            threshold = (clean_center + noisy_center) / 2.0
                            robust_mask = (curr_loss <= threshold).float()

                    l_inter_complete = (curr_loss * robust_mask).sum() / (robust_mask.sum() + 1e-8)
            else:
                if self.prototypes_initialized:
                    C_v, C_u = self.prototypes_view0, self.prototypes_view1
                    q_v = self._soft_assign(z_t[0][complete_idx], C_v, tau=self.proto_tau)
                    q_u = self._soft_assign(z_t[1][complete_idx], C_u, tau=self.proto_tau)
                    gamma_i = self.gamma_global[idx[complete_idx]].unsqueeze(1) if (hasattr(self, 'gamma_global') and idx is not None) else torch.ones(complete_idx.numel(), 1, device=device)

                    p_u_given_v = torch.matmul(q_v, self.R_0to1)
                    p_v_given_u = torch.matmul(q_u, self.R_1to0)

                    z_hat_u = torch.matmul(p_u_given_v, C_u)
                    z_hat_v = torch.matmul(p_v_given_u, C_v)

                    rel_u = (p_u_given_v * q_u).sum(dim=1, keepdim=True)
                    rel_v = (p_v_given_u * q_v).sum(dim=1, keepdim=True)
                    conf_u = gamma_i * rel_u
                    conf_v = gamma_i * rel_v

                    noise_scale = 0.05
                    z_hat_u = z_hat_u + torch.randn_like(z_hat_u) * noise_scale
                    z_hat_v = z_hat_v + torch.randn_like(z_hat_v) * noise_scale

                    tgt_u = conf_u * z_t[1][complete_idx] + (1.0 - conf_u) * z_hat_u
                    tgt_v = conf_v * z_t[0][complete_idx] + (1.0 - conf_v) * z_hat_v

                    max_conf_u = p_u_given_v.max(dim=1).values
                    max_conf_v = p_v_given_u.max(dim=1).values
                    weight_u = conf_u.squeeze(1) + (1.0 - conf_u.squeeze(1)) * max_conf_u
                    weight_v = conf_v.squeeze(1) + (1.0 - conf_v.squeeze(1)) * max_conf_v

                    mp_complete_0 = self.kernel_affinity(z_t[0][complete_idx])
                    mp_complete_1 = self.kernel_affinity(z_t[1][complete_idx])
                    l_inter_0_ps = self.cl(p[0][complete_idx], tgt_u.detach(), mp_complete_1, return_per_sample=True) * weight_u
                    l_inter_1_ps = self.cl(p[1][complete_idx], tgt_v.detach(), mp_complete_0, return_per_sample=True) * weight_v
                    l_inter_complete = (l_inter_0_ps + l_inter_1_ps).mean() / 2
                else:
                    mp_complete_0 = self.kernel_affinity(z_t[0][complete_idx])
                    mp_complete_1 = self.kernel_affinity(z_t[1][complete_idx])
                    l_inter_complete = (
                        self.cl(p[0][complete_idx], z_t[1][complete_idx], mp_complete_1) +
                        self.cl(p[1][complete_idx], z_t[0][complete_idx], mp_complete_0)
                    ) / 2

        # ------------------------------------------------------------------
        # 4) incomplete-sample branch: sparse transported prototype supervision
        # ------------------------------------------------------------------
        l_missing_assign = torch.tensor(0.0, device=device)
        missing_loss_enabled = (
            self.prototypes_initialized and
            (epoch is not None and epoch >= self.missing_start_epoch)
        )

        if missing_loss_enabled:
            terms_assign = []
            eta_vals, alpha_val, peak_vals, entropy_vals = [], [], [], []
            total_missing_cond = 0
            total_missing_keep = 0
            for missing_view in range(self.n_views):
                present_view = 1 - missing_view
                cond = (~mask[:, missing_view]) & mask[:, present_view]
                cond_count = int(cond.sum().item())
                if cond_count == 0:
                    continue
                total_missing_cond += cond_count

                z_present_t = z_t[present_view][cond]
                z_present_o = z[present_view][cond]

                proto_present = self.prototypes_view0 if present_view == 0 else self.prototypes_view1
                proto_missing = self.prototypes_view0 if missing_view == 0 else self.prototypes_view1
                R_present_to_missing = self.R_0to1 if present_view == 0 else self.R_1to0

                q_present = self._soft_assign(z_present_t, proto_present, tau=self.proto_tau)
                q_missing_dense = torch.matmul(q_present, R_present_to_missing)
                q_missing_sparse = self._sparsify_assignment(q_missing_dense, topk=self.sparse_topk, tau=self.sparse_temp)

                z_proto = torch.matmul(q_missing_sparse, proto_missing)
                z_dec_t = self.cross_view_decoder[present_view](z_present_t)
                z_dec_o = self.cross_view_decoder[present_view](z_present_o)

                alpha, eta = self._pseudo_confidence(q_missing_sparse, z_proto, z_dec_t)

                peak = q_missing_sparse.max(dim=1).values
                entropy = self._assignment_entropy(q_missing_sparse)
                eta_vals.append(eta.squeeze(1).detach())
                alpha_val.append(alpha.squeeze(1).detach())
                peak_vals.append(peak.detach())
                entropy_vals.append(entropy.detach())

                keep = (eta.squeeze(1) >= self.min_missing_conf)
                keep_count = int(keep.sum().item())
                total_missing_keep += keep_count
                if keep_count == 0:
                    continue

                z_mix = 0.5 * z_dec_o + 0.5 * z_proto
                pred_assign = self._soft_assign(z_mix[keep], proto_missing, tau=self.proto_tau)
                target_assign = q_missing_sparse[keep].detach()
                assign_loss = F.kl_div((pred_assign + 1e-8).log(), target_assign, reduction='none').sum(dim=1)

                weight = eta[keep].squeeze(1).detach()
                terms_assign.append((assign_loss * weight).sum() / (weight.sum() + 1e-8))

            if len(terms_assign) > 0:
                l_missing_assign = torch.stack(terms_assign).mean()
                debug_info['missing_assign'] = float(l_missing_assign.detach().item())

            if total_missing_cond > 0:
                debug_info['keep_ratio'] = total_missing_keep / float(total_missing_cond)
                if len(eta_vals) > 0:
                    eta_all = torch.cat(eta_vals)
                    alpha_all = torch.cat(alpha_val)
                    peak_all = torch.cat(peak_vals)
                    entropy_all = torch.cat(entropy_vals)
                    debug_info['eta_mean'] = float(eta_all.mean().item())
                    debug_info['alpha_mean'] = float(alpha_all.mean().item())
                    debug_info['eta_max'] = float(eta_all.max().item())
                    debug_info['peak_mean'] = float(peak_all.mean().item())
                    debug_info['entropy_mean'] = float(entropy_all.mean().item())

        # ------------------------------------------------------------------
        # 5) intra-view branch on observed views only
        # ------------------------------------------------------------------
        intra_terms = []
        for i in range(self.n_views):
            if intra_mp[i] is None:
                continue
            intra_terms.append(self.cl(z[i][present_idx[i]], z_t[i][present_idx[i]], intra_mp[i]))
        l_intra = torch.stack(intra_terms).mean() if len(intra_terms) > 0 else torch.tensor(0.0, device=device)

        total_loss = l_intra + l_inter_complete
        total_loss = total_loss + self.missing_loss_weight * l_missing_assign
        self.latest_debug = debug_info
        return total_loss

    def _encode_masked(self, encoder_list, data, mask):
        device = data[0].device
        N = data[0].shape[0]
        out = []
        for i in range(self.n_views):
            feat = torch.zeros(N, self.feature_dim[i], device=device)
            valid = mask[:, i]
            if valid.any():
                feat[valid] = encoder_list[i](data[i][valid])
            out.append(feat)
        return out

    @torch.no_grad()
    def _sparsify_assignment(self, q, topk=3, tau=0.25, eps=1e-12):
        q = q.clamp_min(eps)
        if tau is not None and tau > 0:
            q = q.pow(1.0 / tau)
        q = q / q.sum(dim=1, keepdim=True)

        topk = min(max(1, topk), q.shape[1])
        if topk < q.shape[1]:
            val, ind = torch.topk(q, k=topk, dim=1)
            sparse_q = torch.zeros_like(q)
            sparse_q.scatter_(1, ind, val)
            q = sparse_q
        q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
        return q

    @torch.no_grad()
    def _pseudo_confidence(self, q_sparse, z_proto, z_dec, eps=1e-12):
        peak = q_sparse.max(dim=1, keepdim=True).values
        entropy = -(q_sparse * (q_sparse + eps).log()).sum(dim=1, keepdim=True) / math.log(q_sparse.shape[1])
        agreement = ((L2norm(z_proto) * L2norm(z_dec)).sum(dim=1, keepdim=True) + 1.0) / 2.0
        alpha = (peak * (1.0 - entropy)).clamp(0.0, 1.0)
        eta = (alpha * agreement).clamp(0.0, 1.0)
        return alpha, eta

    @torch.no_grad()
    def _assignment_entropy(self, q_sparse, eps=1e-12):
        return -(q_sparse * (q_sparse + eps).log()).sum(dim=1) / math.log(q_sparse.shape[1])

    def get_debug_state(self):
        return dict(self.latest_debug)

    @torch.no_grad()
    def kernel_affinity(self, z, temperature=0.1, step: int = 5):
        z = L2norm(z)
        G = (2 - 2 * (z @ z.t())).clamp(min=0.)
        G = torch.exp(-G / temperature)
        G = G / G.sum(dim=1, keepdim=True)
        G = torch.matrix_power(G, step)
        alpha = 0.5
        G = torch.eye(G.shape[0], device=z.device) * alpha + G * (1 - alpha)
        return G

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(self.online_encoder[i].parameters(), self.target_encoder[i].parameters()):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    @torch.no_grad()
    def extract_directional_feature(
        self,
        data,
        mask,
        source_view: int,
        include_target_when_source_missing: bool = True,
    ):
        if self.n_views != 2:
            raise NotImplementedError('Directional extraction currently assumes exactly 2 views.')
        target_view = 1 - source_view

        def _build_features(src_view: int, subset_mask: torch.Tensor):
            tgt_view = 1 - src_view
            z_source = self.target_encoder[src_view](data[src_view][subset_mask])
            z_target_dec = self.cross_view_decoder[src_view](z_source)
            eta = None

            if self.prototypes_initialized:
                proto_source = L2norm(self.prototypes_view0 if src_view == 0 else self.prototypes_view1)
                proto_target = L2norm(self.prototypes_view0 if tgt_view == 0 else self.prototypes_view1)
                R_source_to_target = self.R_0to1 if src_view == 0 else self.R_1to0

                assign_source = self._soft_assign(z_source, proto_source, tau=self.proto_tau)
                assign_target = torch.matmul(assign_source, R_source_to_target)
                assign_target = self._sparsify_assignment(assign_target, topk=self.sparse_topk, tau=self.sparse_temp)
                z_proto = torch.matmul(assign_target, proto_target)
                alpha, eta = self._pseudo_confidence(assign_target, z_proto, z_target_dec)
                z_target = alpha * z_proto + (1.0 - alpha) * z_target_dec
            else:
                z_target = z_target_dec
                eta = torch.ones((z_target.shape[0], 1), device=z_target.device, dtype=z_target.dtype)

            out = [None, None]
            out[src_view] = z_source
            out[tgt_view] = z_target
            out = [L2norm(feat) for feat in out]
            feat = torch.cat(out, dim=1)
            return feat, eta

        valid_source = mask[:, source_view]
        valid_target = (
            (~valid_source) & mask[:, target_view]
            if include_target_when_source_missing
            else None
        )

        features = []
        if valid_source.any():
            feat_src, _ = _build_features(source_view, valid_source)
            features.append(feat_src)
        if include_target_when_source_missing and valid_target.any():
            feat_tgt, eta_tgt = _build_features(target_view, valid_target)
            keep = eta_tgt.squeeze(1) >= self.min_missing_conf
            features.append(feat_tgt[keep])
            kept_mask = torch.zeros_like(valid_target)
            kept_idx = valid_target.nonzero(as_tuple=False).squeeze(1)
            kept_mask[kept_idx[keep]] = True
            valid_target = kept_mask

        if len(features) == 0:
            return None, None, None

        return torch.cat(features, dim=0), valid_source, valid_target

    def _soft_assign(self, z, proto, tau=0.1, eps=1e-12):
        z = L2norm(z)
        proto = L2norm(proto)
        logits = (z @ proto.t()) / max(tau, eps)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def compute_and_store_prototypes(self, data_loader, device, epoch=0, seed=0, fp_ratio=0.0, msrt=0.0, data_set_name="dataset"):
        self.eval()

        all_z_target = [[] for _ in range(self.n_views)]
        all_masks = []
        all_idx = []
        all_labels_v0 = []
        all_labels_v1 = []

        for batch in data_loader:
            if len(batch) == 6:
                idx, samples, y0, y1, id1, id2 = batch
                batch_mask = torch.ones((y0.size(0), self.n_views), dtype=torch.bool)
            else:
                idx, samples, mask, y0, y1, id1, id2 = batch
                batch_mask = mask

            all_masks.append(batch_mask.cpu())
            all_idx.append(idx.to(device))
            all_labels_v0.append(y0.cpu())
            all_labels_v1.append(y1.cpu())

            for i in range(self.n_views):
                samples[i] = samples[i].to(device, non_blocking=True)
            batch_mask = batch_mask.to(device)
            z_t = self._encode_masked(self.target_encoder, samples, batch_mask)

            for i in range(self.n_views):
                all_z_target[i].append(z_t[i].cpu())

        all_z_target = [torch.cat(all_z_target[i], dim=0) for i in range(self.n_views)]
        all_masks = torch.cat(all_masks, dim=0)
        all_idx = torch.cat(all_idx, dim=0)
        all_labels_v0 = torch.cat(all_labels_v0, dim=0).numpy()
        all_labels_v1 = torch.cat(all_labels_v1, dim=0).numpy()

        if hasattr(self, 'inter_loss_global'):
            inter_loss_global = self.inter_loss_global[all_idx]
        else:
            inter_loss_global = torch.zeros(all_idx.shape[0], device=device)

        prototypes = []
        for i in range(self.n_views):
            valid_idx = all_masks[:, i] == 1
            valid_features = all_z_target[i][valid_idx].numpy()

            gmm_model = GaussianMixture(
                n_components=self.n_classes, covariance_type="diag",
                random_state=42, init_params="k-means++", n_init=3,
                reg_covar=1e-3, max_iter=100
            )
            valid_features_norm = nn.functional.normalize(torch.from_numpy(valid_features), dim=-1).numpy()
            gmm_model.fit(valid_features_norm)
            proto = torch.from_numpy(gmm_model.means_).float()
            prototypes.append(proto)

        self.prototypes_view0.copy_(prototypes[0].to(device))
        self.prototypes_view1.copy_(prototypes[1].to(device))
        self.prototypes_initialized = True

        self._align_via_sinkhorn_ot(
            prototypes, all_z_target, inter_loss_global, all_masks, all_idx,
            all_labels_v0, all_labels_v1, device, loss_conf_thresh=0.5
        )
        
        self.train()

    @torch.no_grad()
    def _align_via_sinkhorn_ot(self, prototypes, all_z_target, inter_loss, all_masks, all_idx, all_labels_v0, all_labels_v1, device, loss_conf_thresh=0.5):
        K = self.n_classes
        feat0, feat1 = all_z_target[0].to(device), all_z_target[1].to(device)
        N = feat0.shape[0]
        valid_pair_mask = (all_masks[:, 0] & all_masks[:, 1]).to(device)

        valid_loss = inter_loss[valid_pair_mask]
        L = valid_loss.view(-1, 1)
        c1, c2 = L.min(), L.max()
        for _ in range(5):
            d1, d2 = torch.abs(L - c1), torch.abs(L - c2)
            assign = (d1 < d2).float()
            sum_assign = assign.sum()
            if sum_assign > 0:
                c1 = (L * assign).sum() / sum_assign
            sum_unassign = (1 - assign).sum()
            if sum_unassign > 0:
                c2 = (L * (1 - assign)).sum() / sum_unassign
        clean_center, noisy_center = torch.min(c1, c2), torch.max(c1, c2)

        if noisy_center - clean_center < self.loss_distance:
            gamma_valid = torch.ones_like(valid_loss)
        else:
            v_min, v_max = valid_loss.min().item(), valid_loss.max().item()
            valid_loss_norm = (valid_loss - v_min) / (v_max - v_min + 1e-10)
            x = valid_loss_norm.detach().cpu().numpy().reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4, random_state=42, init_params="kmeans", n_init=3)
            gmm.fit(x)
            prob = gmm.predict_proba(x)
            clean_comp = gmm.means_.reshape(-1).argmin()
            gamma_valid = torch.from_numpy(prob[:, clean_comp]).float().to(device)

        gamma = torch.zeros(N, device=device)
        gamma[valid_pair_mask] = gamma_valid

        max_idx = all_idx.max().item()
        if not hasattr(self, 'gamma_global') or self.gamma_global.shape[0] <= max_idx:
            self.register_buffer('gamma_global', torch.zeros(max(max_idx + 1, 50000), device=device))
        self.gamma_global[all_idx] = gamma

        A0 = self._soft_assign(feat0, self.prototypes_view0, tau=self.proto_tau)
        A1 = self._soft_assign(feat1, self.prototypes_view1, tau=self.proto_tau)
        clean_mask = (gamma >= loss_conf_thresh).float()
        w = gamma * clean_mask

        Co = (A0[valid_pair_mask] * w[valid_pair_mask].unsqueeze(1)).t() @ A1[valid_pair_mask]
        cost_0to1 = -Co / (Co.abs().max() + 1e-12)

        p_marginal = (A0[valid_pair_mask].mean(dim=0) + 1e-8)
        q_marginal = (A1[valid_pair_mask].mean(dim=0) + 1e-8)
        p_marginal, q_marginal = p_marginal / p_marginal.sum(), q_marginal / q_marginal.sum()

        pi_0to1 = sinkhorn_log_domain(p_marginal, q_marginal, cost_0to1, reg=0.05, niter=100)
        self.R_0to1.copy_(F.normalize(pi_0to1, p=1, dim=1))
        self.R_1to0.copy_(F.normalize(pi_0to1.t(), p=1, dim=1))


class FCN(nn.Module):
    def __init__(self, dim_layer=None, norm_layer=None, act_layer=None, drop_out=0.0, norm_last_layer=True):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))
        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_hidden),
                                 act_layer(),
                                 nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        x = self.mlp(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None, return_per_sample=False):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        
        if return_per_sample:
            return nll_loss.sum(dim=1)
        
        loss = nll_loss.mean()
        return loss
