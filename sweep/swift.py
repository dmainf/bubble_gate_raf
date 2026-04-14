import sys
import torch
from pathlib import Path

SWIFT_DIR = Path(__file__).parent.parent / "SwiFT" / "project"
sys.path.insert(0, str(SWIFT_DIR))

from module.models.swin4d_transformer_ver7 import SwinTransformer4D as SwinTransformer4D_ver7

CKPT_PATH = Path(__file__).parent.parent / "SwiFT" / "pretrained_models" / "contrastive_pretrained.ckpt"


def load_swift_encoder():
    net = SwinTransformer4D_ver7(
        img_size=(96, 96, 96, 20),
        in_chans=1,
        embed_dim=36,
        window_size=(4, 4, 4, 4),
        first_window_size=(4, 4, 4, 4),
        patch_size=(6, 6, 6, 1),
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        c_multiplier=2,
        last_layer_full_MSA=True,
        to_float=True,
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    net.load_state_dict(sd, strict=False)
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    return net
