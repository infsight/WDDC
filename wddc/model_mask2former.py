import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Optional
import cv2
import numpy as np

# from detectron2.modeling import BACKBONE_REGISTRY
# from .mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
# from .mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor

from ..backbones.clip import *
from ..backbones.utils import tokenize

from .dift_encoder import DIFT


class Difseg(nn.Module):
    def __init__(self,
                 dift_config=dict(projection_dim=[2048, 1024, 512, 256],
                                  projection_dim_x4=256,
                                  model_id="stabilityai/stable-diffusion-xl-base-1.0",
                                  diffusion_mode="inversion",
                                  input_resolution=[512, 512],
                                  prompt="",
                                  negative_prompt="",
                                  guidance_scale=-1,
                                  scheduler_timesteps=[80, 60, 40, 20, 1],
                                  save_timestep=[4, 3, 2, 1, 0],
                                  num_timesteps=5,
                                  idxs=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0],
                                        [3, 1],
                                        [3, 2]]),
                 dift_type='HyperFeature',
                 backbone,
                 text_encoder,
                 decode_head,
                 class_names,
                 context_length,
                 context_decoder=None,
                 token_embed_dim=512, 
                 text_dim=512,
                 neck=None,
                 identity_head=None,
                 visual_reg=True,
                 textual_reg=True,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **args):
        super().__init__()
        
        self.tau = 0.07
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_classes = len(class_names)
        self.context_length = context_length

        if pretrained is not None:
            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = 'pretrained/CLIP-ViT-B-16.pt'
        
        self.backbone = DIFT(dift_config=dift_config, dift_type=dift_type)
        self.text_encoder = builder.build_backbone(text_encoder); self.text_encoder.init_weights()
        self.neck = builder.build_neck(neck) if neck is not None else None
        self.context_decoder = builder.build_backbone(context_decoder) if context_decoder is not None else None

        # requires_grad False
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False

        # build head
        self.decode_head = builder.build_head(decode_head) if decode_head is not None else None
        self.identity_head = builder.build_head(identity_head) if identity_head is not None else None

        # coop
        self.text_encoder.to('cuda')
        prompt_num = self.text_encoder.context_length - self.context_length
        self.texts = torch.cat([tokenize(c, context_length=context_length) for c in class_names]).to('cuda')
        self.contexts = nn.Parameter(torch.randn(1, prompt_num, token_embed_dim))
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        nn.init.trunc_normal_(self.contexts)
        nn.init.trunc_normal_(self.gamma)

    def forward(self, x):
        image_embeddings = self.encoder(x)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder(image_embeddings)
        outputs_class, outputs_mask = self.transformer_decoder(mask_features, transformer_encoder_features, multi_scale_features, query_pos)
        # masks = self.postprocess_masks(
        #         low_res_masks,
        #         input_size=(512,512),
        #         original_size=(512,512),
        #     )
        return outputs_mask

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (512, 512),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


if __name__ == '__main__':
    img = cv2.imread('autodl-tmp/diffusion-segmentation/astronaut_rides_horse.png')

    img = cv2.resize(img, dsize=(512, 512))
    img_tensor = torch.Tensor(img)
    img_tensor = img_tensor.permute(2, 0, 1)

    img_tensor = img_tensor.to(device='cuda', dtype=torch.float)
    img_tensor = img_tensor[None, ...]
    print(img_tensor.shape, img_tensor.dtype)
    
    model = Difseg().to('cuda')
    output = model(img_tensor)
    print(output.shape)