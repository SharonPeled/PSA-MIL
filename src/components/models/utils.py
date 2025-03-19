import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.configs import Configs
import os

def plot_attention_heatmap(attention_scores, rows, cols, inference, slide_uuid, cmap='coolwarm', prefix=''):
    # attention_scores are softmax normalized

    if Configs.get('ATTN_MAP_SAVE_PATH') is None:
        return

    new_stage = 'test' if inference else 'train'

    if Configs.get('ATTN_MAP_SAVE_DICT') is None:
        d = {
            'fold': 0,
            'stage': new_stage,
            'idx': 0
        }
        Configs.set('ATTN_MAP_SAVE_DICT', d)

    d = Configs.get('ATTN_MAP_SAVE_DICT')
    if new_stage != d['stage']:
        d['idx'] = 0
        if new_stage == 'test':
            d['stage'] = 'test'
        elif new_stage == 'train':
            d['stage'] = 'train'
            d['fold'] += 1

    if d['stage'] != 'test':
        return

    path = os.path.join(Configs.get('ATTN_MAP_SAVE_PATH'), f"{d['fold']}", d['stage'], f"{prefix}{d['idx']}_{slide_uuid}.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    d['idx'] += 1



    attention_scores = attention_scores.detach().cpu()

    # Create an empty attention map
    attention_map = torch.zeros(rows.max()+1, cols.max()+1)

    # Aggregate scores into the attention map
    attention_map.index_put_((rows[0], cols[0]), attention_scores[0], accumulate=True)

    # Save to file
    np.save(path, attention_map.numpy())


    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Convert to numpy for visualization
    attention_map_np = attention_map.numpy()

    # Plot the attention map
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_map_np, cmap=cmap, vmin=0, vmax=1)  # Ensures colormap is 0 to 1
    plt.title(f"Min: {attention_scores.min():.2e}, Max: {attention_scores.max():.2e}")
    # plt.colorbar(label="Attention Score")
    # plt.axis('off')
    # plt.show()

    plt.savefig(path.replace('npy', 'png'), bbox_inches='tight')

    print(f'Save attention heatmap: {path}')




def calc_safe_auc(y_true, y_score, **kwargs):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_score, **kwargs)
    except Exception as e:
        print(e)
        return np.nan


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient

    Args:
        - h : torch.Tensor

    Returns:
        - risk : torch.Tensor

    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs







