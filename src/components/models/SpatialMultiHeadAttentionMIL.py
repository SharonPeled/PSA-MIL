import torch
from functools import partial
import torch.nn as nn
from timm.layers import Mlp, DropPath
from timm.models.vision_transformer import LayerScale, Attention
from src.components.models.GatedAttention import ResidualFullyConnected
from src.components.models.utils import MultiInputSequential
import math
from src.components.models.DecayNetwork import DecayNetwork, solve_for_local_k
from src.components.models.DistanceScaler import DistanceScaler
import torch.nn.functional as F
from src.components.models.utils import plot_attention_heatmap


class SpatialMultiHeadAttentionMIL(nn.Module):

    def __init__(self,
                 num_classes,
                 embed_dim,
                 attn_dim,
                 num_heads,
                 depth,
                 num_layers_adapter,
                 patch_drop_rate,
                 qkv_bias,
                 reg_terms,
                 pool_type='cls_token',
                 mlp_ratio=4.,
                 qk_norm=False,
                 proj_drop=0.,
                 drop_rate=0.,
                 attn_drop=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 mlp_layer=Mlp):
        super(SpatialMultiHeadAttentionMIL, self).__init__()
        self.adapter = ResidualFullyConnected(n_channels=embed_dim, m_dim=attn_dim, numLayer_Res=num_layers_adapter)
        self.patch_drop_rate = patch_drop_rate
        self.head = nn.Linear(attn_dim, num_classes)
        self.head_drop = nn.Dropout(drop_rate)
        self.pool_type = pool_type
        self.reg_terms = reg_terms

        if pool_type == 'cls_token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, attn_dim))
            nn.init.normal_(self.cls_token, std=1e-6)

        self.layer_norm = nn.LayerNorm(attn_dim, eps=1e-6)
        self.blocks = MultiInputSequential(*[
            SpatialBlock(
                dim=attn_dim,
                num_heads=num_heads,
                layer_num=layer_num,
                mlp_ratio=mlp_ratio,
                pool_type=pool_type,
                qkv_bias=qkv_bias,
                reg_terms=self.reg_terms,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for layer_num in range(depth)])
        self.num_heads = num_heads
        # self.aux_nets = nn.ModuleList([HeadAuxNet(attn_dim // num_heads, attn_dim // num_heads, num_classes, pool_type)
        #                               for _ in range(self.num_heads)])
        if self.pool_type == 'attention':
            self.attention_pool = nn.Linear(attn_dim, 1)

    def pos_embed(self, x):
        if self.pool_type == 'cls_token':
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return x

    # def patch_drop(self, x, row, col):
    #     num_rows = x.size(1)
    #     num_to_keep = int((1 - self.patch_drop_rate) * num_rows)
    #     indices = torch.randperm(num_rows)[:num_to_keep]
    #     indices = indices[:8000]  # max num tiles
    #     x = x[:, indices]
    #     row = row[:, indices]
    #     col = col[:, indices]
    #     return x, row, col

    def forward_features(self, x, distance, indices, row, col, inference, slide_uuid, logger):
        x = self.adapter(x)
        # x, row, col = self.patch_drop(x, row, col)
        x = self.pos_embed(x)
        x, _, _, _, _, _, lambda_p_list = self.blocks(x, distance, indices, row, col, (inference, slide_uuid, logger))
        x = self.layer_norm(x)
        return x, _, lambda_p_list

    def forward_head(self, x, row, col, inference, slide_uuid):
        if self.pool_type == 'cls_token':
            x = x[:, 0]  # taking cls_tokens
        elif self.pool_type == 'avg':
            x = x.mean(dim=1)
        elif self.pool_type == 'attention':
            attn_scores = self.attention_pool(x).squeeze(-1)  # (batch_size, seq_len)
            # Normalize scores to get attention weights
            attn_weights = attn_scores.softmax(dim=-1)  # (batch_size, seq_len)

            plot_attention_heatmap(attn_weights, row, col, inference, slide_uuid, prefix='attn_pool_')

            # Compute weighted sum of token embeddings
            attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
            weighted_sum = torch.sum(x * attn_weights, dim=1)  # (batch_size, attn_dim)
            x = weighted_sum
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, x, row, col, distance, indices=None, inference=True, slide_uuid=None, logger=None):
        x, _, lambda_p_list = self.forward_features(x, distance, indices, row, col, inference, slide_uuid, logger)
        x = self.forward_head(x, row, col, inference, slide_uuid)
        if inference:
            return x
        # scores_heads, logits_heads = self.get_aux_logits(x_per_head[0])  # TODO: this takes only first head
        return x, None, None, lambda_p_list

    def get_aux_logits(self, x_per_head):
        per_head_logits = []
        per_head_scores = []
        for i in range(self.num_heads):
            x_head = x_per_head[:, i, :, :]  # Shape: (B, N, head_dim)
            score__head = self.aux_nets[i](x_head)
            logits_head = self._get_logits_from_scores(score__head)
            per_head_scores.append(score__head.squeeze(-1))
            per_head_logits.append(logits_head.squeeze(-1))
        res_scores = torch.stack(per_head_scores, dim=0)
        res_logits = torch.stack(per_head_logits, dim=0)
        if self.num_heads == 1:
            res_scores = res_scores.unsqueeze(-1)
            res_logits = res_logits.unsqueeze(-1)
        return res_scores, res_logits

    @property
    def device(self):
        # Determine and return the current device
        return next(self.parameters()).device

    def log_distance_scaler(self, logger):
        for block in self.blocks:
            attn = block.attn
            for h in range(attn.num_heads):
                logger.experiment.log_metric(logger.run_id, f"param_distance_scaler{h}",
                                             attn.distance_scaler.param[h].detach().cpu())
                logger.experiment.log_metric(logger.run_id, f"bias_distance_scaler{h}",
                                             attn.distance_scaler.bias[h].detach().cpu())

    def _get_logits_from_scores(self, scores):
        if len(scores.shape) == 2 and scores.shape[1] > 1:
            logits = F.softmax(scores, dim=-1)
        else:
            logits = F.sigmoid(scores)
        return logits


class SpatialBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            layer_num,
            qkv_bias,
            mlp_ratio=4.,
            pool_type='cls_token',
            qk_norm=False,
            reg_terms={},
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialAttention(
            dim,
            num_heads=num_heads,
            layer_num=layer_num,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            pool_type=pool_type,
            reg_terms=reg_terms
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, distance, indices=None, row=None, col=None, metadata=(), lambda_p_all = None):
        x_attn, _, lambda_p = self.attn(self.norm1(x), distance, indices=indices, row=row, col=col, metadata=metadata)
        x_attn = self.attn.proj(x_attn)
        x_attn = self.attn.proj_drop(x_attn)
        x = x + self.drop_path1(self.ls1(x_attn))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if lambda_p_all is None:
            # print(lambda_p)
            return x, distance, indices, row, col, metadata, lambda_p
        else:
            # print(torch.stack([lambda_p_all, lambda_p], dim=0))
            return x, distance, indices, row, col, metadata, torch.stack([lambda_p_all, lambda_p], dim=0)


class SpatialAttention(Attention):

    def __init__(
            self,
            dim,
            num_heads,
            layer_num,
            qkv_bias,
            pool_type='cls_token',
            reg_terms={},
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)
        self.reg_terms = reg_terms
        self.cls_token_reg = self.reg_terms.get('CLS_TOKEN_REG')
        self.decay_type = self.reg_terms.get('DECAY_TYPE')
        self.is_rpe = self.reg_terms.get('RPE')
        self.local_attention = self.reg_terms.get('LOCAL_ATTENTION')
        self.pool_type = pool_type
        self.num_heads = num_heads
        self.layer_num = layer_num
        self.decay_nn = DecayNetwork(self.decay_type, dim, num_heads, self.reg_terms.get('SLIDE_REPR'),
                                     decay_clip=self.reg_terms.get('DECAY_CLIP'))
        self.distance_scaler = DistanceScaler(self.reg_terms.get('DISTANCE_SCALER'), num_heads)
        self.max_distance = 20
        self.rpe_table = nn.Parameter(torch.randn(self.max_distance+1, self.num_heads))
        self.learnable_sigma = torch.nn.Parameter(torch.tensor(1.0))  # Initial value 1.0

    def add_noise_to_distance(self, distance):
        B, seq_size, _ = distance.shape

        # Calculate the number of cells to add noise to
        num_noised_cells = int(self.cls_token_reg[0] * seq_size)
        random_indices = torch.randperm(seq_size)[:num_noised_cells]
        noise = torch.distributions.Exponential(self.cls_token_reg[1]).sample(num_noised_cells)

        zeros_row = torch.zeros(B, 1, seq_size, device=distance.device)
        zeros_row[:, 1, random_indices] = noise

        zeros_col = torch.zeros(B, seq_size + 1, 1, device=distance.device)
        random_indices += 1 # making sure cell (0,0) remains 0
        zeros_col[:, random_indices, 1] = noise
        return zeros_row, zeros_col

    # def modify_distance_for_class_token(self, distance):
    #     B, seq_size, seq_size = distance.shape
    #     if self.cls_token_reg is None or self.cls_token_reg[0] == 0:
    #         zeros_row = torch.zeros(B, 1, seq_size, device=distance.device)
    #         zeros_col = torch.zeros(B, seq_size + 1, 1, device=distance.device)
    #     else:
    #         zeros_row, zeros_col = self.add_noise_to_distance(distance)
    #     distance = torch.cat([zeros_row, distance], dim=1)
    #     distance = torch.cat([zeros_col, distance], dim=2)
    #     return distance

    def modify_distance_for_class_token(self, distance, indices):
        B, seq_size, seq_size = distance.shape
        if self.cls_token_reg is None or self.cls_token_reg[0] == 0.0:
            zeros_row_dis = torch.zeros(B, 1, seq_size, device=distance.device, dtype=distance.dtype)
            zeros_col_dis = torch.zeros(B, seq_size + 1, 1, device=distance.device, dtype=distance.dtype)
            zeros_row_ind = torch.zeros(B, 1, seq_size, device=distance.device, dtype=indices.dtype)
            zeros_col_ind = torch.zeros(B, seq_size + 1, 1, device=distance.device, dtype=indices.dtype)
        else:
            raise NotImplementedError
            zeros_row, zeros_col = self.add_noise_to_distance(distance)

        distance = torch.cat([zeros_row_dis, distance], dim=1)
        distance = torch.cat([zeros_col_dis, distance], dim=2)

        indices = torch.cat([zeros_row_ind, indices], dim=1)
        indices = torch.cat([zeros_col_ind, indices], dim=2)
        return distance, indices

    def compute_rpe(self, distance):
        B, N, _ = distance.shape  # Batch size (B) and sequence length (N)

        # Clamp the distance to max_distance for indexing purposes
        clamped_distance = torch.clamp(distance, max=self.max_distance).long()  # Ensure integer for indexing

        # Create an empty tensor to store RPE values (B, N, N, num_heads)
        rpe = torch.zeros(B, N, N, self.num_heads, device=distance.device)

        # Mask for distances less than or equal to max_distance
        within_max_mask = (distance <= self.max_distance)

        # Index into RPE table only for distances within max_distance
        rpe[within_max_mask] = self.rpe_table[clamped_distance[within_max_mask]]  # Fill RPE for small distances

        # For distances greater than max_distance, use a single encoding vector
        large_distance_encoding = self.rpe_table[self.max_distance]  # Shared vector for large distances
        rpe[~within_max_mask] = large_distance_encoding  # Apply to masked positions

        # Permute to match attention shape (B, num_heads, N, N)
        rpe = rpe.permute(0, 3, 1, 2)

        return rpe

    def clip_attn_according_to_decay(self, attn_after_softmax, decay):
        mask = (decay == 0)

        # Set the masked attention weights to 0
        attn_after_clipping = torch.where(mask, torch.zeros_like(attn_after_softmax), attn_after_softmax)

        # Calculate the sum of non-zero attention weights along dim=-1
        remaining_sum = attn_after_clipping.sum(dim=-1, keepdim=True)

        # Renormalize the non-zero attention weights so they sum to 1
        normalized_attn = attn_after_clipping / remaining_sum

        return normalized_attn

    def forward(self, x, distance, indices=None, row=None, col=None, metadata=()):
        # print(x.device, distance.device, indices.device)
        B, N, C = x.shape  # Batch size, number of instances, feature dimension

        # Compute Queries, Keys, Values
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Extract Q, K, V

        # if not self.reg_terms.get('RELAX_ASSUMPTION_1'):
        #     # ✅ Assumption 1 is active → Normalize Queries and Keys
        #     q = F.normalize(q, p=2, dim=-1)  # L2 normalize
        #     k = F.normalize(k, p=2, dim=-1)  # L2 normalize
        #     q = q * self.scale  # Scale by sqrt(d_k)
        #
        #     if (self.reg_terms.get('DECAY_CLIP') is not None and self.reg_terms.get('DECAY_CLIP') > 0 and
        #         self.reg_terms.get('DECAY_TYPE') is not None) and self.reg_terms.get('REDUCE_FLOPS_LOCAL_K'):
        #         return self.local_k_flops_reduction_attention_computation(indices, distance, q, k, v, B, N, C)
        #
        #     attn = q @ k.transpose(-2, -1)  # Compute dot-product attention
        #     # print(attn.shape)
        #
        # else:
        #     # 🚀 Assumption 1 is relaxed → Full posterior formulation
        #
        #     # ✅ Assumption 3: Use either fixed or learnable variance
        #     if not self.reg_terms.get('RELAX_ASSUMPTION_3'):
        #         sigma_sq = math.sqrt(self.head_dim)  # Fixed variance (default)
        #     else:
        #         sigma_sq = self.learnable_sigma ** 2  # Learnable variance
        #     # return self.memory_efficient_indexed_attention(q, k, v, sigma_sq, distance, indices, B, N, C)

        if not (self.decay_type is None and (self.is_rpe or self.local_attention)):
            assert self.decay_type is not None
            if self.pool_type == 'cls_token':
                distance, indices = self.modify_distance_for_class_token(distance, indices)

            return self.local_k_flops_reduction_full_posterior_vectorized(indices, distance, q, k, v,
                                                                          math.sqrt(self.head_dim), B, N, C, row, col,
                                                                          metadata=metadata)



            # # Compute squared norms ||q_i||^2 and ||k_j||^2
            # q_norm_sq = torch.sum(q ** 2, dim=-1, keepdim=True)  # Shape: (B, H, N, 1)
            # k_norm_sq = torch.sum(k ** 2, dim=-1, keepdim=True).transpose(-2, -1)  # Shape: (B, H, 1, N)
            #
            # # Compute exponent: exp(q^T k / sigma^2)
            # exp_term = torch.exp(q @ k.transpose(-2, -1) / sigma_sq)  # Shape: (B, H, N, N)
            #
            # # Compute softmax numerator and denominator
            # numerator = exp_term * torch.exp(-0.5 * (q_norm_sq + k_norm_sq) / sigma_sq)
            # denominator = torch.sum(numerator, dim=-1, keepdim=True)  # Sum over keys
            #
            # # Compute posterior probability p(t_j = 1 | q_i)
            # attn = numerator / denominator  # Softmax over key dimension


        attn = q @ k.transpose(-2, -1)  # Compute dot-product attention

        if self.pool_type == 'cls_token':
            distance = self.modify_distance_for_class_token(distance)


        if self.decay_type is not None:
            distance = self.distance_scaler(distance)
            decay, lambda_p = self.decay_nn(distance.to(attn.device), x, coords=None,
                                            decay_clip=self.reg_terms.get('DECAY_CLIP'))  # Compute spatial decay
            decay = decay + 1e-6
            # print(decay.shape, attn.shape)
            attn = attn + decay.log()
        elif self.is_rpe:
            rpe = self.compute_rpe(distance.to(attn.device))
            attn = attn + rpe
        elif self.local_attention is not None:
            attn = self.clip_attn_according_to_decay(attn, (distance.to(attn.device) <= self.local_attention).float())

        if self.reg_terms.get('TEMPERATURE'):
            attn = attn / self.reg_terms.get('TEMPERATURE')

        attn = attn.softmax(dim=-1)

        if self.reg_terms.get('DECAY_CLIP') and self.decay_type is not None:
            attn = self.clip_attn_according_to_decay(attn, decay)

        attn = self.attn_drop(attn)  # (B, num_heads, N, N)
        x = attn @ v  # Shape: (B, num_heads, N, head_dim)

        x = x.transpose(1, 2).reshape(B, N, C)

        return x, None, None  # make sure to work on multi layered

    def memory_efficient_indexed_attention(self, q, k, v, sigma_sq, distance, indices, B, N, C):
        """
        Computes memory-efficient attention per head using only the selected keys.

        Args:
            q: Query tensor, shape (B, H, N, D)
            k: Key tensor, shape (B, H, N, D)
            v: Value tensor, shape (B, H, N, D)
            sigma_sq: Float, scaling factor for Gaussian kernel
            indices_K_h: Tensor (B, H, N, max_K_h), contains selected indices per head

        Returns:
            attn_output: (B, H, N, D), final attention output after applying values.
        """

        B, H, N, D = q.shape  # Batch, Heads, Seq Length, Embedding Dim
        # print(f"q shape: {q.shape} (B=batch, H=heads, N=sequence length, D=embedding dim)")
        # print(f"k shape: {k.shape} (B=batch, H=heads, N=sequence length, D=embedding dim)")
        # print(f"v shape: {v.shape} (B=batch, H=heads, N=sequence length, D=embedding dim)")

        attn_output = torch.zeros((B, H, N, D), device=q.device)  # Final output storage

        local_k = solve_for_local_k(decay_type=self.reg_terms.get('DECAY_TYPE'),
                                    param=self.decay_nn.reparam(self.decay_nn.lambda_p),
                                    decay_clip=self.reg_terms.get('DECAY_CLIP'))
        # local_k = torch.tensor([5, 12], dtype=torch.float64)
        num_elements = torch.round(local_k ** 2)
        print(local_k, num_elements)

        # Compute squared norms once for queries and keys
        q_norm_sq = torch.sum(q ** 2, dim=-1, keepdim=True)  # (B, H, N, 1)
        k_norm_sq = torch.sum(k ** 2, dim=-1, keepdim=True)  # (B, H, N, 1)

        for h in range(H):  # Iterate over heads, since each head has its own indices_K_h
            # print(f"\nProcessing Head {h + 1}/{H}")
            K_h = int(num_elements[h].item())  # Extract K for this head (scalar)

            # Extract relevant indices for this head
            indices_K_h_cpu = indices[:, :, :K_h]  # Shape: (B, N, K_h)
            indices_K_h = indices_K_h_cpu.to(q.device)

            distance_K_h_cpu = distance.gather(2, indices_K_h_cpu)
            distance_K_h = distance_K_h_cpu.to(q.device)  #  (B, N, K_h)


            for i in range(N):  # Iterate over queries to reduce memory
                # print(f"\nProcessing Query {i + 1}/{N}")

                # Extract the specific K_h indices for this (batch, head, query)
                key_indices = indices_K_h[:, i]  # Shape: (B, K_h[h])
                # print(f"key_indices shape: {key_indices.shape} (B=batch, K_h=selected keys per query)")

                # ================== Selecting Keys ==================
                # print("\nSelecting relevant keys:")
                # print(f"k is all keys of shape {k.shape}, we need only head {h}, so we slice: k[:, {h}, :, :]")
                k_per_head = k[:, h, :, :]  # Shape: (B, N, D)
                # print(
                #     f"k_per_head shape after slicing: {k_per_head.shape} (B=batch, N=sequence length, D=embedding dim)")
                #
                # print(
                #     f"key_indices needs to be expanded to match gather dimensions. Original shape: {key_indices.shape}")
                key_indices_expanded = key_indices.unsqueeze(-1).expand(-1, -1, D)
                # print(f"key_indices_expanded shape: {key_indices_expanded.shape} (B=batch, K_h, D=embedding dim)")

                k_selected = torch.gather(k_per_head, dim=-2, index=key_indices_expanded)
                # print(f"k_selected shape: {k_selected.shape} (B=batch, K_h=selected keys, D=embedding dim)")
                assert k_selected.shape == (B, key_indices.shape[1], D), "Mismatch in k_selected shape"

                # ================== Selecting Values ==================
                # print("\nSelecting relevant values (same process as keys):")
                v_per_head = v[:, h, :, :]  # Shape: (B, N, D)
                # print(f"v_per_head shape: {v_per_head.shape}")

                v_selected = torch.gather(v_per_head, dim=-2, index=key_indices_expanded)
                # print(f"v_selected shape: {v_selected.shape} (B=batch, K_h, D=embedding dim)")
                assert v_selected.shape == (B, key_indices.shape[1], D), "Mismatch in v_selected shape"

                # ================== Query Vector ==================
                # print("\nExtracting query vector:")
                q_i = q[:, h, i:i + 1, :]  # (B, 1, D)
                # print(f"q_i shape: {q_i.shape} (B=batch, 1=query token, D=embedding dim)")
                assert q_i.shape == (B, 1, D), "Mismatch in q_i shape"

                # ================== Compute exp(qᵀ k / σ²) ==================
                # print("\nComputing exp(qᵀ k / σ²):")
                # print(f"q_i @ k_selected.transpose(-2, -1), where:")
                # print(f"  q_i shape: {q_i.shape} (B, 1, D)")
                # print(f"  k_selected.transpose(-2, -1) shape: {k_selected.transpose(-2, -1).shape} (B, D, K_h)")

                exp_term_i = torch.exp((q_i @ k_selected.transpose(-2, -1)) / sigma_sq)
                # print(f"exp_term_i shape: {exp_term_i.shape} (B, 1, K_h)")
                assert exp_term_i.shape == (B, 1, key_indices.shape[1]), "Mismatch in exp_term_i shape"

                # ================== Compute Squared Norms ==================
                # print("\nComputing squared norms:")
                k_norm_selected = torch.gather(k_norm_sq[:, h, :, :], dim=-2, index=key_indices.unsqueeze(-1))
                # print(f"k_norm_selected shape: {k_norm_selected.shape} (B, K_h, 1)")
                assert k_norm_selected.shape == (B, key_indices.shape[1], 1), "Mismatch in k_norm_selected shape"

                q_i_norm_sq = q_norm_sq[:, h, i:i + 1, :]
                # print(f"q_i_norm_sq shape: {q_i_norm_sq.shape} (B, 1, 1)")

                # ================== Compute Softmax Numerator ==================
                # print("\nComputing softmax numerator:")
                numerator_i = exp_term_i * torch.exp(-0.5 * (q_i_norm_sq + k_norm_selected) / sigma_sq).transpose(-2, -1)
                # print(f"numerator_i shape: {numerator_i.shape} (B, 1, K_h)")
                # print(torch.exp(-0.5 * (q_i_norm_sq + k_norm_selected) / sigma_sq).shape)
                assert numerator_i.shape == (B, 1, key_indices.shape[1]), "Mismatch in numerator_i shape"

                # 🔹 Step 4: Apply decay correction
                decay, lambda_p = self.decay_nn(distance_K_h[:, i:i+1, :], head_ind=h)  # Compute decay
                decay = decay + 1e-6  # Numerical stability
                # print(decay.shape, distance_k_h.shape, attn.shape)
                numerator_i = numerator_i + decay.log()  # Apply decay in log-space
                # print(attn.shape)

                # ================== Compute Softmax Denominator ==================
                # print("\nComputing softmax denominator:")
                denominator_i = torch.sum(numerator_i, dim=-1, keepdim=True)
                # print(f"denominator_i shape: {denominator_i.shape} (B, 1, 1)")
                assert denominator_i.shape == (B, 1, 1), "Mismatch in denominator_i shape"

                # ================== Compute Final Attention Weights ==================
                # print("\nComputing final attention weights:")
                attn_weights = numerator_i / denominator_i
                # print(f"attn_weights shape: {attn_weights.shape} (B, 1, K_h)")
                assert attn_weights.shape == (B, 1, key_indices.shape[1]), "Mismatch in attn_weights shape"

                # ================== Multiply Attention Weights with Selected Values ==================
                # print("\nMultiplying attention weights with selected values:")
                # print(f"attn_weights shape: {attn_weights.shape} (B, 1, K_h)")
                # print(f"v_selected shape: {v_selected.shape} (B, K_h, D)")

                attn_output[:, h, i, :] = attn_weights @ v_selected  # (B, 1, D)
                # print(f"attn_output[:, {h}, {i}, :].shape: {attn_output[:, h, i, :].shape} (B, D)")

                assert attn_output[:, h, i, :].shape == (B, D), "Mismatch in attn_output shape"

        x = attn_output.transpose(1, 2).reshape(B, N, C)

        return x, None, None  # Make sure to work with multi-layered inputs

        # return attn_output  # (B, H, N, D)


    def memory_efficient_full_posterior(self, q, k, sigma_sq):
        B, H, N, D = q.shape  # Batch, Heads, Seq Length, Embedding Dim
        attn = torch.zeros((B, H, N, N), device=q.device)  # Allocate final result

        # Compute squared norms once (avoid recomputation)
        q_norm_sq = torch.sum(q ** 2, dim=-1, keepdim=True)  # Shape: (B, H, N, 1)
        k_norm_sq = torch.sum(k ** 2, dim=-1, keepdim=True).transpose(-2, -1)  # Shape: (B, H, 1, N)

        for i in range(N):  # Iterate over queries to reduce peak memory
            # Compute only the necessary row of the full matrix
            q_i = q[:, :, i:i + 1, :]  # Shape: (B, H, 1, D)
            q_i_norm_sq = q_norm_sq[:, :, i:i + 1, :]  # Shape: (B, H, 1, 1)

            # Compute exp(qᵀ k / σ²) only for this row
            exp_term_i = torch.exp((q_i @ k.transpose(-2, -1)) / sigma_sq)  # Shape: (B, H, 1, N)

            # Compute numerator row-wise
            numerator_i = exp_term_i * torch.exp(-0.5 * (q_i_norm_sq + k_norm_sq) / sigma_sq)  # (B, H, 1, N)

            # Compute denominator row-wise (sum over keys)
            denominator_i = torch.sum(numerator_i, dim=-1, keepdim=True)  # (B, H, 1, 1)

            # Store result in final attention matrix
            attn[:, :, i:i + 1, :] = numerator_i / denominator_i  # (B, H, 1, N)

        return attn


    def local_k_flops_reduction_attention_computation(self, indices, distance, q, k, v, B, N, C):
        local_k = solve_for_local_k(decay_type=self.reg_terms.get('DECAY_TYPE'),
                                    param=self.decay_nn.reparam(self.decay_nn.lambda_p),
                                    decay_clip=self.reg_terms.get('DECAY_CLIP'))
        # local_k = torch.tensor([5, 12], dtype=torch.float64)
        num_elements = torch.round(local_k ** 2)
        # print(local_k, num_elements)

        B, H, N, d = q.shape  # Batch, Heads, Num Tokens, Head Dimension

        # Initialize result tensor
        x = torch.zeros_like(q)  # Same shape as (B, H, N, d)

        lambda_p_list = []  # Store lambda_p for each head if needed

        # 🔹 Step 2: Loop over heads to process different `local_k` per head
        for h in range(H):
            K_h = int(num_elements[h].item())  # Extract K for this head (scalar)

            # Extract relevant indices for this head
            indices_K_h_cpu = indices[:, :, :K_h]  # Shape: (B, N, K_h)
            indices_K_h = indices_K_h_cpu.to(q.device)

            k_h = k[:, h, :, :] # shape (B, N, d)
            q_h = q[:, h, :, :] # shape (B, N, d)
            v_h = v[:, h, :, :] # shape (B, N, d)

            k_selected = k_h[range(k_h.shape[0]), indices_K_h]  # Shape: (B, N, K_h, d)

            # 🔹 Step 3: Compute dot-product attention for only `K_h` keys
            attn = torch.einsum("bnd,bnkd->bnk", q_h, k_selected)
            # print(attn.shape)

            # print(distance.shape, indices_K_h.shape)
            # 🔹 Step 4: Apply decay correction
            distance_k_h = distance.gather(dim=-1, index=indices_K_h_cpu).to(q.device)  # Shape: (B, N, K_h)
            decay, lambda_p = self.decay_nn(distance_k_h, head_ind=h)  # Compute decay
            decay = decay + 1e-6  # Numerical stability
            # print(decay.shape, distance_k_h.shape, attn.shape)
            attn = attn + decay.log()  # Apply decay in log-space
            # print(attn.shape)

            # 🔹 Step 5: Apply softmax over `K_h`
            attn = F.softmax(attn, dim=-1)  # Shape: (B, N, K_h)

            # Dropout if applicable
            attn = self.attn_drop(attn)  # Shape: (B, N, K_h)

            # Gather only the required values using the same `indices_k_h`
            v_selected = v_h[range(v_h.shape[0]), indices_K_h]  # Shape: (B, N, K_h, d)

            # v_selected = v[:, h, :, :].gather(dim=-2, index=indices_K_h.unsqueeze(-1))  # Shape: (B, N, K_h, d)

            # print(x.dtype, attn.dtype, v_selected.dtype)
            # 🔹 Step 6: Compute weighted sum for values
            x[:, h, :, :] = torch.einsum("bnk,bnkd->bnd", attn.float(), v_selected)  # Shape: (B, N, d)
            # print(x.shape)

            lambda_p_list.append(lambda_p)  # Store lambda_p for each head

            # print(lambda_p)

        lambda_p = torch.stack(lambda_p_list)

        x = x.transpose(1, 2).reshape(B, N, C)

        return x, None, None  # Make sure to work with multi-layered inputs

    def local_k_flops_reduction_full_posterior_vectorized(self, indices, distance, q, k, v, sigma_sq, B,
                                                                         N, C, row, col, metadata):
        local_k = solve_for_local_k(decay_type=self.reg_terms.get('DECAY_TYPE'),
                                    param=self.decay_nn.reparam(self.decay_nn.lambda_p),
                                    decay_clip=self.reg_terms.get('DECAY_CLIP'))
        # local_k = torch.tensor([5, 12], dtype=torch.float64)
        num_elements = torch.round(local_k ** 2)
        # print(local_k, num_elements)

        B, H, N, d = q.shape  # Batch, Heads, Num Tokens, Head Dimension

        # Initialize result tensor
        x = torch.zeros_like(q)  # Same shape as (B, H, N, d)

        lambda_p_list = []  # Store lambda_p for each head if needed

        q_norm_sq = torch.sum(q ** 2, dim=-1, keepdim=True)  # Shape: (B, H, N, 1)
        k_norm_sq = torch.sum(k ** 2, dim=-1, keepdim=True)  # Shape: (B, H, N)

        # 🔹 Step 2: Loop over heads to process different local_k per head
        for h in range(H):
            K_h = int(num_elements[h].item())  # Extract K for this head (scalar)

            # Extract relevant indices for this head
            indices_K_h_cpu = indices[:, :, :K_h]  # Shape: (B, N, K_h)
            indices_K_h = indices_K_h_cpu.to(q.device)

            k_h = k[:, h, :, :]  # shape (B, N, d)
            q_h = q[:, h, :, :]  # shape (B, N, d)
            v_h = v[:, h, :, :]  # shape (B, N, d)

            k_selected = k_h[range(k_h.shape[0]), indices_K_h]  # Shape: (B, N, K_h, d)

            # 🔹 Step 3: Compute dot-product attention for only K_h keys
            attn = torch.einsum("bnd,bnkd->bnk", q_h, k_selected) / sigma_sq  # (B, N, K_h)
            # print(attn.shape)

            # (B, H, N, 1) (B, H, 1, N)
            # (B, N, 1) (B, 1, N)
            # (B, N, 1) (B, N, K_h)
            # print(k_norm_sq.shape)
            # print(indices_K_h.shape)
            selected_k_norm_sq = k_norm_sq[:, h].expand(-1, -1, indices_K_h.shape[-1]).gather(1, indices_K_h)
            # print(selected_k_norm_sq.shape)
            # print(q_norm_sq[:, h].shape)
            # print((q_norm_sq[:, h] + selected_k_norm_sq).shape)
            attn -= 0.5 * (q_norm_sq[:, h] + selected_k_norm_sq) / sigma_sq  # Apply squared norms

            # print(distance.shape, indices_K_h.shape)
            # 🔹 Step 4: Apply decay correction
            distance_k_h = distance.gather(dim=-1, index=indices_K_h_cpu).to(q.device)  # Shape: (B, N, K_h)
            decay, lambda_p = self.decay_nn(distance_k_h, head_ind=h)  # Compute decay
            decay = decay + 1e-6  # Numerical stability
            # print(decay.shape, distance_k_h.shape, attn.shape)
            attn = attn + decay.log()  # Apply decay in log-space
            # print(attn.shape)

            # 🔹 Step 5: Apply softmax over K_h
            attn = F.softmax(attn, dim=-1)  # Shape: (B, N, K_h)

            # Dropout if applicable
            attn = self.attn_drop(attn)  # Shape: (B, N, K_h)

            # Gather only the required values using the same indices_k_h
            v_selected = v_h[range(v_h.shape[0]), indices_K_h]  # Shape: (B, N, K_h, d)

            # v_selected = v[:, h, :, :].gather(dim=-2, index=indices_K_h.unsqueeze(-1))  # Shape: (B, N, K_h, d)

            # print(x.dtype, attn.dtype, v_selected.dtype)
            # 🔹 Step 6: Compute weighted sum for values
            x[:, h, :, :] = torch.einsum("bnk,bnkd->bnd", attn.float(), v_selected)  # Shape: (B, N, d)
            # print(x.shape)

            lambda_p_list.append(lambda_p)  # Store lambda_p for each head


            if self.reg_terms.get('COMPUTE_ATTN_AND_SIM'):
                attn_cpu = attn.detach().cpu()
                full_attn = torch.zeros((B, N, N), device=attn_cpu.device, dtype=attn_cpu.dtype)
                # Scatter the local attention values into the full attention matrix
                full_attn.scatter_(-1, indices_K_h_cpu, attn_cpu)
                token_attn = full_attn.mean(dim=1)  # B,N
                plot_attention_heatmap(token_attn,
                                       row,
                                       col,
                                       inference=metadata[0], slide_uuid=metadata[1],
                                       prefix=f'head{h}_')
        lambda_p = torch.stack(lambda_p_list)

        if self.reg_terms.get('COMPUTE_ATTN_AND_SIM') and metadata[2] is not None:
            pairwise_sim = compute_head_similarity(x)[0] # to reduce the B dim
            # print(pairwise_sim)
            for i in range(self.num_heads):
                for j in range(i + 1, self.num_heads):
                    metadata[2].experiment.log_metric(metadata[2].run_id,
                                                                   f"sim_{i}{j}", pairwise_sim[i,j])

        x = x.transpose(1, 2).reshape(B, N, C)

        return x, None, None  # Make sure to work with multi-layered inputs


# # Compute squared norms ||q_i||^2 and ||k_j||^2
#             q_norm_sq = torch.sum(q ** 2, dim=-1, keepdim=True)  # Shape: (B, H, N, 1)
#             k_norm_sq = torch.sum(k ** 2, dim=-1, keepdim=True).transpose(-2, -1)  # Shape: (B, H, 1, N)
            #
            # # Compute exponent: exp(q^T k / sigma^2)
            # exp_term = torch.exp(q @ k.transpose(-2, -1) / sigma_sq)  # Shape: (B, H, N, N)
            #
            # # Compute softmax numerator and denominator
            # numerator = exp_term * torch.exp(-0.5 * (q_norm_sq + k_norm_sq) / sigma_sq)
            # denominator = torch.sum(numerator, dim=-1, keepdim=True)  # Sum over keys
            #
            # # Compute posterior probability p(t_j = 1 | q_i)
            # attn = numerator / denominator  # Softmax over key dimension

import torch


def compute_head_similarity(x):
    """
    Computes the similarity between attention head representations using L2 norm.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, N, d),
                          where B is batch size, H is number of heads,
                          N is the number of tokens, and d is embedding dimension per head.

    Returns:
        torch.Tensor: A similarity matrix of shape (B, H, H) where each entry (h1, h2)
                      represents the mean L2 norm between head representations.
    """
    B, H, N, d = x.shape  # Extract dimensions

    # Compute pairwise L2 distances between heads
    # Expand x to (B, H, 1, N, d) and (B, 1, H, N, d) to compute pairwise distances
    x1 = x.unsqueeze(2)  # (B, H, 1, N, d)
    x2 = x.unsqueeze(1)  # (B, 1, H, N, d)

    # Compute L2 norm across the feature dimension (d)
    l2_dist = torch.norm(x1 - x2, dim=-1)  # (B, H, H, N)

    # Average over all tokens to get a (B, H, H) similarity matrix
    head_similarity = l2_dist.mean(dim=-1)  # (B, H, H)

    return 1 / head_similarity

