# src/timm_unet.py (Definitive Final Version 3)

import torch
import torch.nn as nn
import timm
from monai.networks.blocks import SubpixelUpsample

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # This MONAI block takes `in_channels` and outputs `in_channels` at 2x spatial resolution.
        self.upsample = SubpixelUpsample(spatial_dims=2, in_channels=in_channels, scale_factor=2)
        
        # --- THIS IS THE CORRECTED FORMULA ---
        # The number of input channels for the next convolution is the number of upsampled channels
        # (which is the same as in_channels) plus the number of channels from the skip connection.
        conv_in_channels = in_channels + skip_channels
        
        self.conv1 = ConvBnAct(conv_in_channels, out_channels)
        self.conv2 = ConvBnAct(out_channels, out_channels)
        self.attention = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x

# The rest of the TimmUnet class remains the same as the last version I sent.
# I am including it here again for completeness.
class TimmUnet(nn.Module):
    def __init__(self, backbone_name='caformer_b36.sail_in22k_ft_in1k', pretrained=True):
        super().__init__()
        
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=3, 
            num_classes=0,
            drop_path_rate=0.1,
        )

        self.entry_conv = nn.Sequential(
            nn.Conv2d(5, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        encoder_channels = self.encoder.feature_info.channels()
        decoder_channels = (256, 128, 64, 32, 16)
        
        self.decoder_blocks = nn.ModuleList()
        in_ch = encoder_channels[-1]
        skip_chs = encoder_channels[:-1][::-1]

        for i, out_ch in enumerate(decoder_channels):
            skip_ch = skip_chs[i] if i < len(skip_chs) else 0
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        
        self.head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        if not self.training: # TTA
            x_flipped = torch.flip(x, dims=[-1])
            output_orig = self._forward_features(x)
            output_flipped = self._forward_features(x_flipped)
            return (output_orig + torch.flip(output_flipped, dims=[-1])) / 2
        
        return self._forward_features(x)

    def _forward_features(self, x):
        # Resize input to a "safe" size divisible by 32
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = self.entry_conv(x)
        features = self.encoder(x)
        features.reverse()

        x = features[0]
        skips = features[1:]
        
        for i, block in enumerate(self.decoder_blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
            
        x = self.head(x)
        x = nn.functional.interpolate(x, size=(70, 70), mode='bilinear', align_corners=False)
        return self.final_activation(x)