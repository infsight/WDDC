import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Optional
import cv2
import numpy as np

from .dift_encoder import DIFT
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer


# stabilityai/stable-diffusion-xl-base-1.0
# stabilityai/stable-diffusion-2
class Difseg(nn.Module):
    def __init__(self,
                 dift_config=dict(projection_dim=[2048, 1024, 512, 256],
                                  projection_dim_x4=256,
                                  model_id="stabilityai/stable-diffusion-2-base",
                                  diffusion_mode="inversion",
                                  input_resolution=[512, 512],
                                  prompt="",
                                  negative_prompt="",
                                  guidance_scale=-1,
                                  scheduler_timesteps=[80, 60, 40, 20, 1],
                                  save_timestep=[4, 3, 2, 1, 0],
                                  num_timesteps=5,
                                  # scheduler_timesteps=[20, 1],
                                  # save_timestep=[1, 0],
                                  # num_timesteps=2,
                                  # scheduler_timesteps=[180, 160, 140, 120, 100, 80, 60, 40, 20, 1],
                                  # save_timestep=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                                  # num_timesteps=10,
                                  idxs=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0],
                                        [3, 1],
                                        [3, 2]]),
                 dift_type='HyperFeature'):
        super().__init__()
        self.encoder = DIFT(dift_config=dift_config, dift_type=dift_type)
        
        prompt_embed_dim = 256
        self.decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.image_embedding_size = (64,64)
        embed_dim = prompt_embed_dim
        self.embed_dim = embed_dim
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, x):
        bs = x.shape[0]
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=x.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        image_embeddings = self.encoder(x)
        b_size = image_embeddings.shape[0]
        
        low_res_masks, iou_predictions = self.decoder(
                image_embeddings=image_embeddings,
                image_pe=self.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        masks = self.postprocess_masks(
                low_res_masks,
                input_size=(512,512),
                original_size=(512,512),
            )
        return masks

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


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


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