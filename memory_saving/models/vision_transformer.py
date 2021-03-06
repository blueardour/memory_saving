""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

import memory_saving as ms
import numpy as np

from memory_saving.clip import find_clip_mmse

__all__ = [
    'deit_ms_tiny_patch16_224',
    ]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


import os
import matplotlib.pyplot as plt
import seaborn as sns, numpy as np

random_colors = np.random.rand(12, 3).tolist()

save_root = '/data1/cvpr2022/plots/activation_dist'

colors = [
(0.43, 0.83, 0., 0.3),
(0.97, 0.71, 0., 0.3),
(0.2, 0.77, 1.0, 0.3),
]

def plot_act_dist(name, output):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    plt.clf()
    ax = sns.histplot(output.mean(dim=0).flatten().detach().cpu().numpy(), element="step", kde=False)
    img_name = f"{name}.png"
    plt.title(f"{name}")
    plt.xlabel('Activations')
    plt.savefig(os.path.join(save_root, img_name))

def plot_act_dist_per_head(name, output):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    plt.clf()
    fig, ax = plt.subplots(dpi=150)

    if len(output.shape) == 4:
        B, H, N, _ = output.shape
        data = output
        for i in range(H):
            attn = data[:, i, ...].mean(dim=0).flatten().detach().cpu().numpy()
            # sns.histplot(attn, element="step", kde=False, color=colors[i], ax=ax, label='head-'+str(i))
            sns.histplot(attn, element="step", kde=False, color=(*random_colors[i], 0.3), ax=ax)
    else:
        groups = 3
        B, N, C = output.shape
        data = output.reshape(B, N, groups, C//groups).permute(0,2,1,3)
        for i in range(groups):
            attn = data[:, i, ...].mean(dim=0).flatten().detach().cpu().numpy()
            # sns.histplot(attn, element="step", kde=False, color=colors[i], ax=ax, label='head-'+str(i))
            sns.histplot(attn, element="step", kde=False, color=(*random_colors[i], 0.3), ax=ax)

    plt.title(f"{name}")
    plt.xlabel('Activations')
    plt.legend()
    img_name = f"{name}.png"
    # plt.savefig(os.path.join(save_root, img_name))
    plt.show()
    # print('...')

class AnalyseMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=ms.GELU, drop=0., num_heads=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ms.Linear(in_features, hidden_features, groups=num_heads)
        self.act = act_layer(groups=num_heads)
        self.fc2 = ms.Linear(hidden_features, out_features, groups=num_heads)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        plot_act_dist_per_head(f'{self.name}-in', x)

        plot_act_dist_per_head(f'{self.name}-before-fc1', x)
        x = self.fc1(x)

        plot_act_dist_per_head(f'{self.name}-before-gelu', x)
        x = self.act(x)

        x = self.drop(x)

        plot_act_dist_per_head(f'{self.name}-before-fc2', x)
        x = self.fc2(x)

        x = self.drop(x)

        plot_act_dist_per_head(f'{self.name}-out', x)
        return x

class AnalyseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = ms.Linear(dim, dim * 3, bias=qkv_bias, groups=num_heads)
        self.softmax = ms.Softmax(dim=-1, groups=num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = ms.Linear(dim, dim, groups=num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mm1 = ms.MatMul(groups=num_heads)
        self.mm2 = ms.MatMul(groups=num_heads)

    def forward(self, x):
        B, N, C = x.shape
        plot_act_dist_per_head(f'{self.name}-before-qkv', x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        plot_act_dist_per_head(self.name + '-before-q@k-q', q)
        plot_act_dist_per_head(self.name + '-before-q@k-k', k)
        attn = self.mm1(q, k.transpose(-2, -1)) * self.scale

        plot_act_dist_per_head(f'{self.name}-before-softmax', attn)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        plot_act_dist_per_head(f'{self.name}-before-attn@v-attn', attn)
        plot_act_dist_per_head(f'{self.name}-before-attn@v-v', v)
        x = self.mm2(attn, v).transpose(1, 2).reshape(B, N, C)

        plot_act_dist_per_head(f'{self.name}-before-proj', x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AnalyseBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=ms.GELU, norm_layer=ms.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, groups=num_heads)
        self.attn = AnalyseAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, groups=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = AnalyseMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, num_heads=num_heads)

    def forward(self, x):
        plot_act_dist_per_head(f'{self.name}-before-attn', x)
        residual_1 = self.drop_path(self.attn(self.norm1(x)))

        plot_act_dist_per_head(f'{self.name}-before-res-add-1', residual_1)
        x = x + residual_1

        plot_act_dist_per_head(f'{self.name}-before-mlp', x)
        residual_1 = self.drop_path(self.mlp(self.norm2(x)))

        plot_act_dist_per_head(f'{self.name}-before-res-add-2', residual_1)
        x = x + residual_1

        plot_act_dist_per_head(f'{self.name}-out', x)
        return x

class AnalysePatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        plot_act_dist_per_head(f'{self.name}-in', x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        plot_act_dist_per_head(f'{self.name}-out', x)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=ms.GELU, drop=0., num_heads=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ms.Linear(in_features, hidden_features, groups=num_heads)
        self.act = act_layer(groups=num_heads)
        self.fc2 = ms.Linear(hidden_features, out_features, groups=num_heads)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = ms.Linear(dim, dim * 3, bias=qkv_bias, groups=num_heads)

        # self.q_norm = ms.LayerNorm(head_dim, groups=num_heads)
        # self.k_norm = ms.LayerNorm(head_dim, groups=num_heads)
        # self.v_norm = ms.LayerNorm(head_dim, groups=num_heads)

        self.softmax = ms.Softmax(dim=-1, groups=num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = ms.Linear(dim, dim, groups=num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mm1 = ms.MatMul(groups=num_heads)
        self.mm2 = ms.MatMul(groups=num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # q = self.q_norm(q)
        # k = self.k_norm(k)
        # v = self.v_norm(v)

        # attn = self.mm1(q, k.transpose(-2, -1)) * self.scale
        q = q * self.scale
        attn = self.mm1(q, k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.mm2(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=ms.GELU, norm_layer=ms.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, groups=num_heads)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, groups=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, num_heads=num_heads)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = ms.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class AnalyseViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=ms.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = AnalysePatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            AnalyseBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim, groups=num_heads)

        # Classifier head
        self.head = ms.Linear(embed_dim, num_classes, groups=num_heads) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = ms.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        plot_act_dist_per_head(f'{self.name}-in', x)
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 0]
        plot_act_dist(f'{self.name}-out', x)
        return x

    def forward(self, x):
        plot_act_dist_per_head(f'vit-in', x)
        x = self.forward_features(x)
        x = self.head(x)
        plot_act_dist(f'vit-out', x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=ms.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim, groups=num_heads)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = ms.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = ms.Linear(embed_dim, num_classes, groups=num_heads) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = ms.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def deit_ms_tiny_analyse(pretrained=False, **kwargs):
    model = AnalyseViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(ms.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_ms_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(ms.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_ms_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(ms.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_ms_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(ms.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_ms_small_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


#@register_model
#def vit_base_patch16_224(pretrained=False, **kwargs):
#    model = VisionTransformer(
#        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    model.default_cfg = default_cfgs['vit_base_patch16_224']
#    if pretrained:
#        load_pretrained(
#            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
#    return model
#
#
#@register_model
#def vit_base_patch16_384(pretrained=False, **kwargs):
#    model = VisionTransformer(
#        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    model.default_cfg = default_cfgs['vit_base_patch16_384']
#    if pretrained:
#        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#    return model
#
#
#@register_model
#def vit_base_patch32_384(pretrained=False, **kwargs):
#    model = VisionTransformer(
#        img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    model.default_cfg = default_cfgs['vit_base_patch32_384']
#    if pretrained:
#        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#    return model
#
#
#@register_model
#def vit_large_patch16_224(pretrained=False, **kwargs):
#    model = VisionTransformer(
#        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    model.default_cfg = default_cfgs['vit_large_patch16_224']
#    if pretrained:
#        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#    return model
#
#
#@register_model
#def vit_large_patch16_384(pretrained=False, **kwargs):
#    model = VisionTransformer(
#        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
#        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    model.default_cfg = default_cfgs['vit_large_patch16_384']
#    if pretrained:
#        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#    return model
#
#
#@register_model
#def vit_large_patch32_384(pretrained=False, **kwargs):
#    model = VisionTransformer(
#        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
#        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    model.default_cfg = default_cfgs['vit_large_patch32_384']
#    if pretrained:
#        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#    return model
#
#
#@register_model
#def vit_huge_patch16_224(pretrained=False, **kwargs):
#    model = VisionTransformer(patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
#    model.default_cfg = default_cfgs['vit_huge_patch16_224']
#    return model
#
#
#@register_model
#def vit_huge_patch32_384(pretrained=False, **kwargs):
#    model = VisionTransformer(
#        img_size=384, patch_size=32, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
#    model.default_cfg = default_cfgs['vit_huge_patch32_384']
#    return model
#
#
#@register_model
#def vit_small_resnet26d_224(pretrained=False, **kwargs):
#    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
#    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
#    model = VisionTransformer(
#        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
#    model.default_cfg = default_cfgs['vit_small_resnet26d_224']
#    return model
#
#
#@register_model
#def vit_small_resnet50d_s3_224(pretrained=False, **kwargs):
#    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
#    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[3])
#    model = VisionTransformer(
#        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
#    model.default_cfg = default_cfgs['vit_small_resnet50d_s3_224']
#    return model
#
#
#@register_model
#def vit_base_resnet26d_224(pretrained=False, **kwargs):
#    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
#    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
#    model = VisionTransformer(
#        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
#    model.default_cfg = default_cfgs['vit_base_resnet26d_224']
#    return model
#
#
#@register_model
#def vit_base_resnet50d_224(pretrained=False, **kwargs):
#    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
#    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
#    model = VisionTransformer(
#        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
#    model.default_cfg = default_cfgs['vit_base_resnet50d_224']
#    return model
