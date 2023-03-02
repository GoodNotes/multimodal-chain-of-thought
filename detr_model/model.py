import torch
from torch import nn

from detr_model.backbone import Backbone, Joiner
from detr_model.fb_transformer import Transformer
from detr_model.position_encoding import PositionEmbeddingSine
from detr_model.utils import NestedTensor, nested_tensor_from_tensor_list


class DETREmbedder(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries=100):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        return hs


def get_pretrained_detr_embedder():
    hidden_dim = 256
    backbone = Backbone("resnet101", train_backbone=True, return_interm_layers=False, dilation=True)
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=False)

    detr_embedder = DETREmbedder(backbone_with_pos_enc, transformer, num_queries=100)
    checkpoint = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth", map_location="cpu", check_hash=True
    )

    state_dict = checkpoint["model"]
    state_dict.pop("class_embed.weight", None)
    state_dict.pop("class_embed.bias", None)
    state_dict.pop("bbox_embed.layers.0.weight", None)
    state_dict.pop("bbox_embed.layers.0.bias", None)
    state_dict.pop("bbox_embed.layers.1.weight", None)
    state_dict.pop("bbox_embed.layers.1.bias", None)
    state_dict.pop("bbox_embed.layers.2.weight", None)
    state_dict.pop("bbox_embed.layers.2.bias", None)

    detr_embedder.load_state_dict(checkpoint["model"])
    detr_embedder.eval()
    return detr_embedder

