import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ------------------------------
# Building Blocks
# ------------------------------
class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU (x2) with optional dropout."""

    def __init__(self, in_ch, out_ch, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionBlock(nn.Module):
    """Attention Gate as described in Attention U-Net paper."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Ensure spatial dimensions match for addition
        if g1.shape[-2:] != x1.shape[-2:]:
            # Interpolate g1 to match x1's spatial dimensions
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Ensure psi matches x's spatial dimensions for multiplication
        if psi.shape[-2:] != x.shape[-2:]:
            psi = F.interpolate(psi, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return x * psi


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionBlock(F_g=out_ch, F_l=skip_ch, F_int=out_ch // 2)
        self.conv = ConvBlock(in_ch=out_ch + skip_ch, out_ch=out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            skip = self.attn(x, skip)
            
            # Ensure spatial dimensions match for concatenation (handles progressive resizing)
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# ------------------------------
# Attention U-Net with ResNet34 encoder
# ------------------------------
class AttentionUNet(nn.Module):
    """U-Net with ResNet34 encoder and attention gates."""

    def __init__(self, n_classes: int = 6, dropout: float = 0.1, pretrained: bool = True, 
                 gradient_checkpointing: bool = False):
        super().__init__()
        enc = models.resnet34(pretrained=pretrained)
        
        # Store config for summary
        self.n_classes = n_classes
        self.dropout = dropout
        self.gradient_checkpointing = gradient_checkpointing

        # Encoder layers we will use as feature maps
        self.enc0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # 64, /2
        self.pool0 = enc.maxpool  # /4

        self.enc1 = enc.layer1  # 64, /4
        self.enc2 = enc.layer2  # 128, /8
        self.enc3 = enc.layer3  # 256, /16
        self.enc4 = enc.layer4  # 512, /32

        # Decoder
        self.up4 = UpBlock(512, 256, 256, dropout)
        self.up3 = UpBlock(256, 128, 128, dropout)
        self.up2 = UpBlock(128, 64, 64, dropout)
        self.up1 = UpBlock(64, 64, 64, dropout)  # skip from enc0

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        # initialize weights for decoder
        self._init_weights()
        
        # Enable gradient checkpointing if requested (saves memory at cost of speed)
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _init_weights(self):
        for m in [self.up4, self.up3, self.up2, self.up1, self.final_conv]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        try:
            from torch.utils.checkpoint import checkpoint
            self._checkpoint = checkpoint
            print("[AttentionUNet] Gradient checkpointing enabled for memory efficiency.")
        except ImportError:
            print("[AttentionUNet] Warning: gradient checkpointing not available.")
            self.gradient_checkpointing = False

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x)
        else:
            return self._forward_normal(x)
    
    def _forward_normal(self, x):
        """Standard forward pass."""
        # Store original size for progressive resizing compatibility
        original_size = x.shape[-2:]
        
        # Encoder
        e0 = self.enc0(x)        # (B,64,H/2,W/2)
        e0p = self.pool0(e0)     # (B,64,H/4,W/4)
        e1 = self.enc1(e0p)      # (B,64,H/4,W/4)
        e2 = self.enc2(e1)       # (B,128,H/8,W/8)
        e3 = self.enc3(e2)       # (B,256,H/16,W/16)
        e4 = self.enc4(e3)       # (B,512,H/32,W/32)

        # Decoder
        d4 = self.up4(e4, e3)    # (B,256,H/16,W/16)
        d3 = self.up3(d4, e2)    # (B,128,H/8,W/8)
        d2 = self.up2(d3, e1)    # (B,64,H/4,W/4)
        d1 = self.up1(d2, e0)    # (B,64,H/2,W/2)

        # Use original input size instead of hardcoded scale_factor for progressive resizing compatibility
        out = F.interpolate(d1, size=original_size, mode='bilinear', align_corners=False)  # (B,64,H,W)
        out = self.final_conv(out)  # (B,n_classes,H,W)
        return out
    
    def _forward_with_checkpointing(self, x):
        """Forward pass with gradient checkpointing for memory efficiency."""
        # Store original size for progressive resizing compatibility
        original_size = x.shape[-2:]
        
        # Encoder with checkpointing
        e0 = self._checkpoint(self.enc0, x, use_reentrant=False)
        e0p = self.pool0(e0)
        e1 = self._checkpoint(self.enc1, e0p, use_reentrant=False)
        e2 = self._checkpoint(self.enc2, e1, use_reentrant=False)
        e3 = self._checkpoint(self.enc3, e2, use_reentrant=False)
        e4 = self._checkpoint(self.enc4, e3, use_reentrant=False)

        # Decoder with checkpointing
        d4 = self._checkpoint(self.up4, e4, e3, use_reentrant=False)
        d3 = self._checkpoint(self.up3, d4, e2, use_reentrant=False)
        d2 = self._checkpoint(self.up2, d3, e1, use_reentrant=False)
        d1 = self._checkpoint(self.up1, d2, e0, use_reentrant=False)

        # Use original input size instead of hardcoded scale_factor for progressive resizing compatibility
        out = F.interpolate(d1, size=original_size, mode='bilinear', align_corners=False)
        out = self.final_conv(out)
        return out

    # --------------------------------------------------
    def summary(self, input_size=(3, 256, 256)):
        """Return a miniature summary and parameter count."""
        num_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        try:
            from torchinfo import summary as torchinfo_summary
            summ = torchinfo_summary(self, input_size=(1, *input_size), verbose=0)
            return summ, num_params, trainable
        except ImportError:
            return f"Model parameters: total={num_params}, trainable={trainable}", num_params, trainable 