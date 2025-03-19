import torch
from src.components.objects.Logger import Logger
from src.components.models.AbstractMILClassifier import AbstractMILClassifier
from src.components.models.SpatialMultiHeadAttentionMIL import SpatialMultiHeadAttentionMIL


class PSAClassifier(AbstractMILClassifier):
    def __init__(self, num_classes, task, embed_dim, attn_dim, num_heads, depth, num_layers_adapter, patch_drop_rate,
                 qkv_bias, learning_rate_pairs, weight_decay_pairs, weighting_strategy,
                 pool_type, reg_terms):
        super(PSAClassifier, self).__init__(num_classes, task, learning_rate_pairs, weight_decay_pairs,
                                            weighting_strategy)
        self.spatial_mil = SpatialMultiHeadAttentionMIL(self.num_classes, embed_dim, attn_dim, num_heads, depth,
                                                        num_layers_adapter, patch_drop_rate, qkv_bias,
                                                        pool_type=pool_type,
                                                        reg_terms=reg_terms)
        self.reg_terms = reg_terms
        Logger.log(f"""PSAClassifier created with learnable_distance decay: {reg_terms}.""",
                   log_importance=1)

    def configure_optimizers(self):
        # Get the parameters of model.spatial_mil.blocks[0].attn.decay_nn
        scaled_lr_params = sum([list(self.spatial_mil.blocks[i].attn.decay_nn.parameters())
                            for i in range(len(self.spatial_mil.blocks))], [])
        # Get all other parameters except those in decay_nn
        other_params = []
        for param in self.parameters():
            if any([param is p for p in scaled_lr_params]):
                continue
            other_params.append(param)

        lr_scale = self.reg_terms.get('DECAY_LR_SCALE') if self.reg_terms.get('DECAY_LR_SCALE') else 1
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': self.init_lr},
            {'params': scaled_lr_params, 'lr': self.init_lr * lr_scale},
        ])
        return optimizer

    def on_before_optimizer_step(self, optimizer, optimizer_idx=None):
        super().on_before_optimizer_step(optimizer, optimizer_idx=None)
        opt = self.optimizers(use_pl_optimizer=False)

        if not isinstance(opt, list):
            self.logger.experiment.log_metric(self.logger.run_id, f"scaled_lr", opt.param_groups[1]['lr'])
        else:
            for opt in self.optimizers(use_pl_optimizer=False):
                self.logger.experiment.log_metric(self.logger.run_id, f"scaled_lr", opt.param_groups[1]['lr'])

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     super(PSAClassifier, self).on_train_batch_end(outputs, batch, batch_idx)
    #     lr_scale = self.reg_terms.get('DECAY_LR_SCALE') if self.reg_terms.get('DECAY_LR_SCALE') else 1
    #     wd_scale = self.reg_terms.get('DECAY_WD_SCALE') if self.reg_terms.get('DECAY_WD_SCALE') else 1
    #
    #     opt = self.optimizers(use_pl_optimizer=False)
    #
    #     if not isinstance(opt, list):
    #         opt.param_groups[1]['lr'] *= lr_scale
    #         opt.param_groups[1]['weight_decay'] *= wd_scale
    #     else:
    #         for opt in self.optimizers(use_pl_optimizer=False):
    #             opt.param_groups[1]['lr'] *= lr_scale
    #             opt.param_groups[1]['weight_decay'] *= wd_scale


    # def on_train_start(self) -> Optional[int]:
    #     if self.log_flops:
    #         from torchinfo import summary
    #         device = self.device
    #         with torch.no_grad():
    #             self.spatial_mil.to('cpu')
    #             input_data = [torch.rand(1, 4975, 384), torch.randint(0,50, (1, 4975)), torch.randint(0,50, (1, 4975)),
    #                           torch.rand(1, 4975, 4975)]
    #             model_summary = summary(
    #                 self.spatial_mil,
    #                 input_data=input_data,
    #                 col_names=["output_size", "num_params", "mult_adds"],
    #                 verbose=0
    #             )
    #             self.logger.experiment.log_metric(self.logger.run_id, "flops", model_summary.total_mult_adds)
    #             Logger.log(f"""FLOPS logged: {model_summary.total_mult_adds:,}""", log_importance=1)
    #             trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #             Logger.log(f'trainable_params {trainable_params:,}', log_importance=1)
    #             self.to(device)
    #             self.log_flops = False
    #     super().on_train_start()

    def _forward(self, batch, inference=True):
        x_embed, slide_uuids, rows, cols, distance_batch, indices_batch = batch[0], batch[3], batch[6], batch[7], batch[8], batch[9]
        scores_list = []
        # due to different slide sizes
        for i in range(len(x_embed)):
            x_bag, slide_uuid, row, col, distance_bag = x_embed[i], slide_uuids[i], rows[i], cols[i], distance_batch[i]
            indices = indices_batch[i]
            scores_list.append(self.spatial_mil(x_bag.unsqueeze(0), row.unsqueeze(0),
                               col.unsqueeze(0), torch.from_numpy(distance_bag).unsqueeze(0), torch.from_numpy(indices).unsqueeze(0),
                                                inference=inference, slide_uuid=slide_uuid, logger=self.logger))
            # Logger.log(f"""{i}, {scores_list}""",
            #            log_importance=1)
        if inference:
            scores = torch.stack([score.squeeze(0) for score in scores_list])
            return scores
        scores = torch.stack([score.squeeze(0) for score, _, _, _ in scores_list])
        # x_per_head_scores = torch.stack([x_per_head_score.squeeze(0) for _, x_per_head_score, _, _ in scores_list])
        # x_per_head_logits = torch.stack([x_per_head_logit.squeeze(0) for _, _, x_per_head_logit, _ in scores_list])
        # lambda_p = torch.stack([lambda_p_list for _, _, _, lambda_p_list in scores_list])
        return scores, None, None, None

    def forward(self, batch, inference=True):
        """
        :param batch is a list of params (x, c, path ...)
        In this way each class could require its own forward parameters
        :return:
        """
        if inference:
            return super(PSAClassifier, self).forward(batch)
        scores, aux_scores, aux_logits, lambda_p = self._forward(batch, inference=inference)
        return scores, self._get_logits_from_scores(scores), aux_scores, aux_logits, lambda_p

    def logging_distance_decays(self, lambda_p_B_layered, metric_str):
        for layer_ind in range(lambda_p_B_layered.shape[0]):
            for head_ind in range(lambda_p_B_layered.shape[1]):
                # print(f"metric_str_{layer_ind}_{head_ind}", lambda_p_B_layered[layer_ind, head_ind])
                self.logger.experiment.log_metric(self.logger.run_id, f"{metric_str}_{layer_ind}_{head_ind}",
                                                  lambda_p_B_layered[layer_ind, head_ind])

    def training_step(self, batch, batch_idx):
        self.update_warmup(batch_idx)

        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]

        x, y, path = batch[0], batch[2], batch[5]

        scores, logits, aux_scores, aux_logits, lambda_p = self.forward(batch, inference=False)

        loss = self.loss(scores, logits, y, path, aux_scores, aux_logits, lambda_p)

        self.logger.experiment.log_metric(self.logger.run_id, "train_loss", loss.detach().cpu())
        return loss

    def entropy_loss_gaussian_binning(self, rates):
        """
        Computes entropy loss based on Gaussian soft binning.

        Args:
            rates (torch.Tensor): Tensor of shape (L, H) with values between 0 and 1.

        Returns:
            torch.Tensor: Mean entropy loss.
        """
        L, H = rates.shape  # Get dimensions

        # Define bin centers (equally spaced between 0 and 1)
        bin_centers = torch.linspace(0, 1, steps=H, device=rates.device)  # Shape: (H,)
        # print(bin_centers)

        # Define Gaussian spread based on bin width
        sigma = 0.1 # Bin range (adaptive to H)

        # Compute soft binning probabilities using Gaussian similarity
        rates_expanded = rates.unsqueeze(-1)  # Shape: (L, H, 1)
        bin_probs = torch.exp(-((rates_expanded - bin_centers) ** 2) / (2 * sigma ** 2))  # Gaussian kernel

        # Normalize to get valid probability distributions
        bin_probs = bin_probs / bin_probs.sum(dim=-1, keepdim=True)  # Shape: (L, H, H)
        # print(bin_probs)
        # Compute mean probability distribution over H
        probabilities = bin_probs.mean(dim=1)  # Shape: (L, H)
        # print(probabilities)

        # Compute entropy
        epsilon = 1e-10  # Small value to avoid log(0)
        entropy_per_L = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=1)  # Shape: (L,)

        # Return mean entropy as loss
        return entropy_per_L.mean()  # Scalar loss

    def kde_entropy_loss(self, rates, bandwidth=0.1, num_samples=1000):
        """
        Computes entropy loss using Kernel Density Estimation (KDE) with Monte Carlo sampling.

        Args:
            rates (torch.Tensor): Tensor of shape (N,), where N is the number of theta values.
            bandwidth (float): Bandwidth for KDE smoothing.
            num_samples (int): Number of Monte Carlo samples.

        Returns:
            torch.Tensor: Estimated entropy loss (higher means more diversity).
        """
        # print(rates)
        N = rates.shape[0]  # Number of theta values

        if N < 2:
            # If there's only one theta value, entropy is zero (no diversity).
            return torch.tensor(0.0, device=rates.device)

        # Step 1: Monte Carlo Sampling from KDE
        sampled_rates = rates[torch.randint(0, N, (num_samples,))]  # Sample existing values
        noise = torch.randn_like(sampled_rates) * bandwidth  # Add Gaussian noise
        samples = sampled_rates + noise  # Synthetic samples from KDE

        # Step 2: Compute KDE density estimates at sampled points
        pairwise_sq_dists = (samples.unsqueeze(0) - rates.unsqueeze(1)) ** 2  # Shape: (N, num_samples)
        kernel_values = torch.exp(-pairwise_sq_dists / (2 * bandwidth ** 2)) / (
                    torch.sqrt(torch.tensor(2 * torch.pi)) * bandwidth)

        # Estimate p(rates) at sampled points
        p_rates = kernel_values.mean(dim=0)  # Shape: (num_samples,)

        # Step 3: Compute entropy approximation
        entropy_loss = -torch.mean(torch.log(p_rates + 1e-10))  # Avoid log(0) with small epsilon

        return entropy_loss

    def loss(self, scores, logits, y, path, aux_scores, aux_logits, lambda_p):
        if self.reg_terms['DECAY_TYPE'] is not None:
            rates = torch.stack([block.attn.decay_nn.get_params('rates') for block in self.spatial_mil.blocks])
            thetas = torch.stack([block.attn.decay_nn.get_params('thetas') for block in self.spatial_mil.blocks])
            local_Ks = torch.stack([block.attn.decay_nn.get_params('local_Ks') for block in self.spatial_mil.blocks])

            if rates.shape == torch.Size([1]):
                rates = rates.reshape(1,1)
                thetas = rates.reshape(1,1)
                local_Ks = rates.reshape(1,1)
            if len(rates.shape) == 1:
                rates = rates.unsqueeze(0)
                thetas = rates.unsqueeze(0)
                local_Ks = rates.unsqueeze(0)
            # new shape: [num_layers, num_heads]
            # print('rates', rates.shape, rates)
            # print('local_Ks', local_Ks.shape, local_Ks)
            # print('thetas', thetas.shape, thetas)
            # print('\n')


            self.logging_distance_decays(rates, 'rates')
            self.logging_distance_decays(thetas, 'thetas')
            self.logging_distance_decays(local_Ks, 'local_Ks')
            # print(rates, thetas, local_Ks)


        task_loss = super().loss(scores, logits, y, path)  # Compute the prediction task loss
        self.logger.experiment.log_metric(self.logger.run_id, "task_loss", task_loss.detach().cpu())

        if self.reg_terms.get('BETA') is not None and self.reg_terms.get('BETA') > 0:
            aux_loss = torch.mean(torch.stack([super(PSAClassifier, self).loss(aux_scores[:, i, :], aux_logits[:, i, :], y, path)
                                               for i in range(self.spatial_mil.num_heads)]))
            self.logger.experiment.log_metric(self.logger.run_id, "aux_loss", aux_loss.detach().cpu())
            task_loss = task_loss + aux_loss*self.reg_terms.get('BETA')

        if self.reg_terms.get('ALPHA') is not None and self.reg_terms.get('ALPHA') > 0:
            # div_loss = self.entropy_loss_gaussian_binning(rates)
            div_loss = self.kde_entropy_loss(rates.squeeze(0))
            self.logger.experiment.log_metric(self.logger.run_id, "div_loss", div_loss.detach().cpu())
            # print(div_loss)
            task_loss = task_loss - div_loss*self.reg_terms.get('ALPHA')
            # task_loss =  - div_loss*self.reg_terms.get('ALPHA')


        return task_loss




