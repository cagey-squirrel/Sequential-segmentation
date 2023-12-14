from __future__ import absolute_import
import torch
import my_segmentation_models_pytorch as smp


class SmpUnet(torch.nn.Module):


    def __init__(self, encoder, in_channels, out_channels, weights, attention=None, do_mutual_attention=True, device=None):
        super().__init__()
        self.model = smp.Unet(encoder_name=encoder, in_channels=in_channels, classes=out_channels, encoder_weights=weights, activation='sigmoid', decoder_attention_type=attention, do_mutual_attention=do_mutual_attention, device=device)


    def forward(self, images):
        return self.model(images) 