import torch.nn as nn
from timm.layers import Mlp, DropPath
from timm.models.vision_transformer import LayerScale, Attention
from src.components.models.GatedAttention import ResidualFullyConnected
from src.components.models.utils import MultiInputSequential
import math
from src.components.models.DecayNetwork import DecayNetwork, solve_for_local_k
import torch.nn.functional as F
import torch
import numpy as np


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
        attn_dim = attn_dim * num_heads
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
    #     x = x[:, indices]
    #     row = row[:, indices]
    #     col = col[:, indices]
    #     return x, row, col

    def forward_features(self, x, distance, indices, row, col, inference, slide_uuid, logger):
        x = self.adapter(x)
        # x, row, col = self.patch_drop(x, row, col)
        x = self.pos_embed(x)
        x, _, _, _, _, _ = self.blocks(x, distance, indices, row, col, (inference, slide_uuid, logger))
        x = self.layer_norm(x)
        return x

    def forward_head(self, x, row, col, inference, slide_uuid):
        if self.pool_type == 'cls_token':
            x = x[:, 0]  # taking cls_tokens
        elif self.pool_type == 'avg':
            x = x.mean(dim=1)
        elif self.pool_type == 'attention':
            attn_scores = self.attention_pool(x).squeeze(-1)  # (batch_size, seq_len)
            # Normalize scores to get attention weights
            attn_weights = attn_scores.softmax(dim=-1)  # (batch_size, seq_len)
            # Compute weighted sum of token embeddings
            attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
            weighted_sum = torch.sum(x * attn_weights, dim=1)  # (batch_size, attn_dim)
            x = weighted_sum
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, x, row, col, distance, indices=None, inference=True, slide_uuid=None, logger=None):
        x = self.forward_features(x, distance, indices, row, col, inference, slide_uuid, logger)
        x = self.forward_head(x, row, col, inference, slide_uuid)
        return x

    @property
    def device(self):
        # Determine and return the current device
        return next(self.parameters()).device


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

    def forward(self, x, distance, indices=None, row=None, col=None, metadata=()):
        x_attn = self.attn(self.norm1(x), distance, indices=indices, row=row, col=col, metadata=metadata)
        x_attn = self.attn.proj(x_attn)
        x_attn = self.attn.proj_drop(x_attn)
        x = x + self.drop_path1(self.ls1(x_attn))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, distance, indices, row, col, metadata


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
        self.decay_type = self.reg_terms.get('DECAY_TYPE')
        self.pool_type = pool_type
        self.num_heads = num_heads
        self.layer_num = layer_num
        self.decay_nn = DecayNetwork(self.decay_type, num_heads, decay_clip=self.reg_terms.get('DECAY_CLIP'))

    def modify_distance_for_class_token(self, distance, indices):
        B, seq_size, seq_size = distance.shape

        zeros_row_dis = torch.zeros(B, 1, seq_size, device=distance.device, dtype=distance.dtype)
        zeros_col_dis = torch.zeros(B, seq_size + 1, 1, device=distance.device, dtype=distance.dtype)
        zeros_row_ind = torch.zeros(B, 1, seq_size, device=distance.device, dtype=indices.dtype)
        zeros_col_ind = torch.zeros(B, seq_size + 1, 1, device=distance.device, dtype=indices.dtype)

        distance = torch.cat([zeros_row_dis, distance], dim=1)
        distance = torch.cat([zeros_col_dis, distance], dim=2)

        indices = torch.cat([zeros_row_ind, indices], dim=1)
        indices = torch.cat([zeros_col_ind, indices], dim=2)
        return distance, indices

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
        B, N, C = x.shape  # Batch size, number of instances, feature dimension

        # Compute Queries, Keys, Values
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Extract Q, K, V


        if self.pool_type == 'cls_token':
            distance, indices = self.modify_distance_for_class_token(distance, indices)

        return self.local_k_flops_reduction_full_posterior_vectorized(indices, distance, q, k, v,
                                                                      math.sqrt(self.head_dim), B, N, C, row, col,
                                                                      metadata=metadata)

    def local_k_flops_reduction_full_posterior_vectorized(self, indices, distance, q, k, v, sigma_sq, B,
                                                                         N, C, row, col, metadata):
        """
        This implementation of the full posterior computation with decay priors and sptial pruning.
        Basically, computing the local K emerges for each head (basedo on its decay and tau parameter), and then
        iterating and summing up the posterior according to the final equation in the paper.
        This was implemented this way to reduce the FLOPs count while still being vectorized (the entire vectorized approach
        could not leverage the FLOP reduction of the locality of heads)
        :param indices:  for each instance, the rest of instances indices sorted by distance (so taking the n closest elements would be fast)
        :param distance: pairwise distance matrix
        :param q:
        :param k:
        :param v:
        :param sigma_sq:
        :param B:
        :param N:
        :param C:
        :param row:
        :param col:
        :param metadata:
        :return:
        """

        local_k = solve_for_local_k(decay_type=self.reg_terms.get('DECAY_TYPE'),
                                    param=self.decay_nn.reparam(self.decay_nn.lambda_p),
                                    decay_clip=self.reg_terms.get('DECAY_CLIP'))
        num_elements = torch.round(np.pi * local_k ** 2)

        B, H, N, d = q.shape  # Batch, Heads, Num Tokens, Head Dimension

        # Initialize result tensor
        x = torch.zeros_like(q)  # Same shape as (B, H, N, d)

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

            selected_k_norm_sq = k_norm_sq[:, h].expand(-1, -1, indices_K_h.shape[-1]).gather(1, indices_K_h)

            attn -= 0.5 * (q_norm_sq[:, h] + selected_k_norm_sq) / sigma_sq  # Apply squared norms

            # 🔹 Step 4: Apply decay correction
            distance_k_h = distance.gather(dim=-1, index=indices_K_h_cpu).to(q.device)  # Shape: (B, N, K_h)
            decay = self.decay_nn(distance_k_h, head_ind=h)  # Compute decay
            decay = decay + 1e-6  # Numerical stability
            attn = attn + decay.log()  # Apply decay in log-space

            # 🔹 Step 5: Apply softmax over K_h
            attn = F.softmax(attn, dim=-1)  # Shape: (B, N, K_h)

            # Dropout if applicable
            attn = self.attn_drop(attn)  # Shape: (B, N, K_h)

            # Gather only the required values using the same indices_k_h
            v_selected = v_h[range(v_h.shape[0]), indices_K_h]  # Shape: (B, N, K_h, d)

            # 🔹 Step 6: Compute weighted sum for values
            x[:, h, :, :] = torch.einsum("bnk,bnkd->bnd", attn.float(), v_selected)  # Shape: (B, N, d)

        x = x.transpose(1, 2).reshape(B, N, C)

        return x


