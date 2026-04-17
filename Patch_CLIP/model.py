"""
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from resnet import SimpleDownsample
import torch.nn.functional as F
from auxillary import *
ENABLE_ATTN_MAP = 0

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn_select = ENABLE_ATTN_MAP

        if self.attn_select == 0:
            self.attn = nn.MultiheadAttention(d_model, n_head) #change 01 nn.Multi for visualization
        else:
            self.attn = MultiheadAttention(d_model, n_head) #change 01 nn.Multi for visualization
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.attn_probs = None
        self.attn_grad = None

    def set_attn_probs(self, attn_probs):
        self.attn_probs = attn_probs

    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if self.attn_select==0:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0] 
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, attention_probs_forward_hook=self.set_attn_probs,
                         attention_probs_backwards_hook=self.set_attn_grad)[0] #for visualization , gradients are enabled 

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.grid_size = (input_resolution // patch_size)
        self.width = width
  
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND #50, B, width
        x = self.transformer(x)
        patch_x = x[1:, :, :] #49, B, width
        x = x.permute(1, 0, 2)  # LND -> NLD #B, 50, width

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, patch_x

import pdb

class SparsePatchEmbeddingTransformer(nn.Module):
    def __init__(self, input_size=1024, hidden_size=2048, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_size))
        encoder_layers = nn.TransformerEncoderLayer( d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder( encoder_layer=encoder_layers, num_layers=num_layers)
        #self.sparse_pos_embed = nn.Parameter()
        
    def forward(self, x, pos_x=None):
        x = x.unsqueeze(0)  # x = 1 x 1024 x n
        x = x.permute(2,0,1) # x = n x 1 x 1024
        if pos_x is not None:
            x = x + pos_x.unsqueeze(1) # nx1x1024
        bs = x.size(0)
        x = torch.cat([self.cls_token.expand(1 ,-1, -1), x], dim=0) #n+1,1,2048
        x = self.transformer_encoder(x)
        x = x[0, :, :] #only take the cls_token learnt embedding x= 1 x 1024
        x = F.normalize(x, p=2, dim=-1)
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length
        self.num_patches = (image_resolution // vision_patch_size) ** 2
        self.width = vision_width
        self.output_dim = embed_dim

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features, local_image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features/ local_image_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    
class ClipWrapper(nn.Module):
    def __init__(self, local_clip, concept_name, input_resolution, clip_input_resolution):
        super().__init__()

        self.text_pair = ["{}".format(concept_name), "no {}".format(concept_name)]
        self.concept_name = concept_name

        self.local_clip = local_clip
        self.conv_downsample = SimpleDownsample(in_channels=3, input_resolution=input_resolution, out_resolution=clip_input_resolution) 
        self.concept_proj = nn.ModuleList([nn.Linear(self.local_clip.num_patches, 1, bias=False), 
                                           nn.Linear(self.local_clip.width, self.local_clip.output_dim, bias=False),
                                           ]) 
        self.sparsePatchEmbed = SparsePatchEmbeddingTransformer()

        self.concept_proj[0].weight = nn.Parameter(torch.ones(1,self.local_clip.num_patches)/self.local_clip.num_patches) #initialize with average...  
        nn.init.xavier_uniform_(self.concept_proj[0].weight)
        # self.concept_proj[2].weight = nn.Parameter(torch.ones(1,self.local_clip.num_patches)/self.local_clip.num_patches) #initialize with average...  
        # nn.init.xavier_uniform_(self.concept_proj[2].weight)

    def forward(self, image, text, patch_labels, isTrain=True):
            image = self.conv_downsample(image).type(self.local_clip.dtype) # shape = [*, 3, 256, 256]
            global_image_features, local_image_features = self.local_clip.encode_image(image)
            text_features = self.local_clip.encode_text(text)

            local_image_features = local_image_features.permute(1, 2, 0) #B, width, 49
            local_image_features_combined = F.gelu(self.concept_proj[0](local_image_features).type(self.local_clip.dtype)).squeeze(-1) #B, dim, 1
            local_image_features_combined = F.gelu(self.concept_proj[1](local_image_features_combined).type(self.local_clip.dtype))
            #copy 
            # sparse_local_image_features = local_image_features.clone()
            sparse_local_image_features_combined_list = []
            B, w, PNUM = local_image_features.shape
            # cosine similarity as logits
            logit_scale = self.local_clip.logit_scale.exp()

            #set some of the sparse patch embeds to zero based on patch number for each image in the batch
            if isTrain:
              for bs in range(B): #for each image 
                current_patch_list = []
                for k in range(len(patch_labels)):
                    if patch_labels[k][bs] != 0:
                        current_patch_list.append(patch_labels[k][bs].item())
                # mask = torch.ones(PNUM, dtype=torch.bool)
                # mask[current_patch_list] = False
                # sparse_local_image_features[bs, :, mask]= 0
                current_patch_list.append(0)
                sparse_local_image_features = local_image_features[bs, :, current_patch_list]
                current_patch_list = [idx + 1 if idx != 0 else 0 for idx in current_patch_list]
                sparse_image_posEnc = self.local_clip.visual.positional_embedding[current_patch_list, :]
                #sparse_features_combined = self.sparsePatchEmbed(sparse_local_image_features, None).type(self.local_clip.dtype)
                sparse_features_combined = self.sparsePatchEmbed(sparse_local_image_features, sparse_image_posEnc).type(self.local_clip.dtype)
                sparse_features_combined = F.gelu(self.concept_proj[1](sparse_features_combined).type(self.local_clip.dtype))
                sparse_local_image_features_combined_list.append(sparse_features_combined)

            else: #during evaluate we dont use the patch labels 
              for bs in range(B): #for each image 
                current_patch_list = []
                for k in range(PNUM):
                    temp_patch_embed = F.gelu(self.concept_proj[1](local_image_features[bs,:, k]).type(self.local_clip.dtype))
                    temp_patch_embed = temp_patch_embed /temp_patch_embed.norm(dim=-1, keepdim=True)
                    #correlate to text and check if logit > 0.5
                    temp_patch_logit = logit_scale * temp_patch_embed @ text_features.t()
                    #Softmax it for the two text features
                    sm_patch_logit = F.softmax(temp_patch_logit, -1)[0]                    
                    if sm_patch_logit > 0.50: 
                        #set the sparse embedding to zero 
                        # sparse_local_image_features[bs,:,k] = 0
                        current_patch_list.append(k)
                current_patch_list.append(0)
                sparse_local_image_features = local_image_features[bs, :, current_patch_list]
                current_patch_list = [idx + 1 if idx != 0 else 0 for idx in current_patch_list]
                sparse_image_posEnc = self.local_clip.visual.positional_embedding[current_patch_list, :]
                sparse_features_combined = self.sparsePatchEmbed(sparse_local_image_features, sparse_image_posEnc).type(self.local_clip.dtype)
                sparse_features_combined = F.gelu(self.concept_proj[1](sparse_features_combined).type(self.local_clip.dtype))
                sparse_local_image_features_combined_list.append(sparse_features_combined)                        

            # sparse_local_image_features_combined = F.gelu(self.concept_proj[2](sparse_local_image_features).type(self.local_clip.dtype)).squeeze(-1) #B, dim, 1
            sparse_local_image_features_combined = torch.stack(sparse_local_image_features_combined_list).squeeze(1)

            p_image_features = []
            for ii in range(self.local_clip.num_patches): #TODO change this hack
                tmp2 = F.gelu(self.concept_proj[1](local_image_features[:,:,ii]).type(self.local_clip.dtype))           
            # we are going to check each patch with the pos and neg text embedding
                tmp2 = tmp2 / tmp2.norm(dim=-1, keepdim=True) # make sure the norm is fine
                p_image_features.append(tmp2) 

            # normalized features
            global_image_features = global_image_features / global_image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            local_image_features_combined = local_image_features_combined/ local_image_features_combined.norm(dim=-1, keepdim=True)
            sparse_local_image_features_combined = sparse_local_image_features_combined/ sparse_local_image_features_combined.norm(dim=-1, keepdim=True)


            logits_per_image = logit_scale * global_image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ global_image_features.t()
            logits_from_patch = logit_scale * local_image_features_combined @ text_features.t()
            sparse_logits_from_patch = logit_scale * sparse_local_image_features_combined @ text_features.t()

            logits_from_each_patch = []
            for ii in range(len(p_image_features)):
                logits_from_each_patch.append(logit_scale * p_image_features[ii] @ text_features.t())  # 49 x B x 2

            return logits_per_image, logits_per_text, logits_from_patch, sparse_logits_from_patch, logits_from_each_patch


class LocalConceptLoss(nn.Module):
    def __init__(self):
        super().__init__()

        #self.weight = torch.tensor([1.0, 5.0])
        self.criterion = nn.CrossEntropyLoss(reduction='sum').to('cuda:0')
        self.global_wt = 1.0
        self.local_weight = 1.0
        self.patch_weight = 1.0
        self.sparse_weight = 0.5
        

    def forward(self, logits_per_image, local_logits_per_image, sparse_local_logits_per_image, p_logits_per_image, labels, patch_labels, device): # B x 2, [Bx2, Bx2,...], [B] (0,1,2,3) 0 == No, 1 yes L, 2 yes R, 3 yes LR
        #labels - (0,1,2,3) 0 == No, 1 yes L, 2 yes R, 3 yes LR
        #patch_label  = (0...49)
        batch = labels.shape[0]
        lbl_01 = labels.clone()
        lbl_01[lbl_01>0] = 1
        lbl_01 = 1-lbl_01
      
        loss_global = self.criterion(logits_per_image, lbl_01)
        loss_local = self.criterion(local_logits_per_image, lbl_01)
        sparse_loss_local = self.criterion(sparse_local_logits_per_image, lbl_01)

        loss_patches = 0
        for i in range(batch):  # for each image     
              for j in range(len(p_logits_per_image)): # for each patch adjust the correct loss
                loss_patches += self.criterion(p_logits_per_image[j][i:(i+1),:],torch.tensor([1], device=device)) 
                # add "not effusion" to all, subtract only for those with Effusion label  
                #since most patches are not effusion anyways 
                for k in range(len(patch_labels)): 
                    patch_num = patch_labels[k][i] #patch_labels is 4x32 shape, TODO fix this shape. probably in append operation. 
                    if j == patch_num and j>0: # current patch matches the labeled patch and ignore 0 as it was padded value
                        loss_patches -= self.criterion(p_logits_per_image[j][i:(i+1),:], torch.tensor([1], device=device)) #subtract the incorrect corr to noE
                        loss_patches += self.criterion(p_logits_per_image[j][i:(i+1),:], torch.tensor([0], device=device)) #add the loss of effusion correlation
                           

        loss_global /= batch
        loss_local /= batch
        loss_patches /= batch
        sparse_loss_local /= batch

        loss = self.global_wt * loss_global + self.local_weight * loss_local + self.patch_weight * loss_patches + self.sparse_weight * sparse_loss_local

        return loss, loss_global, loss_local, loss_patches
    
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()

def load_concept_model(model_path, concept_name, context_length=77, input_resolution=1024, clip_input_resolution=320):
    #load Base model
    state_dict_base = torch.load('../../data/checkpoints/best_64_0.0001_original_35000_0.864.pt')
    local_clip = build_model(state_dict_base)

    #load saved checkpoint which is a ClipWrapper, but save as ClipWrapper_LRHead
    model = ClipWrapper_LRHead(local_clip, concept_name, input_resolution, clip_input_resolution)
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()
    convert_weights(model)

    # 1. filter out unnecessary keys , take only common keys of local_clip
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict #TODO check if this is needed
    #model_dict.update(pretrained_dict) 

    # 3. load the new state dict
    model.load_state_dict(pretrained_dict, strict=False)

    return model

import clip
def load_pretrained_model(model_path, concept_name, context_length=77, input_resolution=1024, clip_input_resolution=320):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    
    concept_model = ClipWrapper(model, concept_name, input_resolution=input_resolution, clip_input_resolution=clip_input_resolution)
    checkpoint = torch.load(model_path, map_location=device)

    # Filter out unnecessary keys based on the layers present in ClipWrapper
    state_dict = {k: v for k, v in checkpoint.items() if k in concept_model.state_dict()}

    # Load the filtered state dictionary into the ClipWrapper instance
    concept_model.load_state_dict(state_dict, strict=False)
    #concept_model.load_state_dict(torch.load(model_path, map_location=device))

    return concept_model
