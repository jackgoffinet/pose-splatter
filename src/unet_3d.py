"""
A 3D Unet autoencoder to encode the input distribution.

"""
__date__ = "December 2024"


import torch
import torch.nn as nn


@torch.no_grad()
def init_unet_primary_skip(
    model,
    in_channels=4,
    small_scale=1e-4,
):
    """
    Custom initialization that:
      - Makes encoder1 & decoder1 close to identity (for first in_channels).
      - Sets final_conv to route those same channels to the first in_channels of output.
      - Minimizes contributions from all other blocks (encoder2..5, decoder2..4, MLP, upconvs) by setting their weights ~0.
      - Sets BatchNorm to identity.
    """

    for name, module in model.named_modules():
        # 1) Handle BatchNorm: set gamma=1, beta=0
        if isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
            continue

        # 2) Handle Linear layers (the MLP in the middle)
        if isinstance(module, nn.Linear):
            # We'll init them to near-zero so they have minimal effect early on
            # or a smaller-than-usual Kaiming or Xavier.
            nn.init.normal_(module.weight, mean=0.0, std=small_scale)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
            continue

        # 3) For all Conv3d and ConvTranspose3d layers, we have different strategies:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            # We'll parse the "name" to see which block it's in.
            
            if 'encoder1' in name or 'decoder1' in name or 'final_conv' in name:
                out_c, in_c, kd, kh, kw = module.weight.shape
                
                # Zero everything first:
                module.weight.zero_()
                # partial identity
                # For encoder1's first conv: in_c == in_channels? or 4 vs out_c=8
                # Do partial identity along the diagonal for min(in_channels, in_c, out_c)
                diag_len = min(in_channels, in_c, out_c)
                for i in range(diag_len):
                    module.weight[i, i, kd//2, kh//2, kw//2] = 1.0
                
                # Also set a small random init for the rest so it’s not strictly zero:
                mask = (module.weight != 0)
                nn.init.normal_(module.weight, mean=0.0, std=small_scale)
                module.weight[mask] = 1.0  # preserve our diagonal "1"
                
                # Zero bias
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            
            else:
                # Near zero init for everything else
                nn.init.normal_(module.weight, mean=0.0, std=small_scale)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    


class Unet3D(nn.Module):
    
    def __init__(
            self,
            in_channels=4,
            out_channels=9,
            base_filters=8,
            z_dim=512,
            rank=32,
            input_size=[80,80,48],
        ):
        super(Unet3D, self).__init__()
        
        self.rank = rank
        contraction_factor = 16
        for s in input_size:
            assert s % contraction_factor == 0
        self.input_size = input_size
        self.ns = [i // 16 for i in input_size]
        self.n_prod = self.ns[0] * self.ns[1] * self.ns[2]

        self.encoder1 = self._conv_block(in_channels, base_filters)
        self.encoder2 = self._conv_block(base_filters, base_filters * 2)
        self.encoder3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.encoder4 = self._conv_block(base_filters * 4, base_filters * 8)
        self.encoder5 = self._conv_block(base_filters * 8, base_filters * 16)

        self.mlp_1 = nn.Sequential(
            nn.Linear(base_filters * 16 * self.n_prod, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim),
        )

        self.mlp_2 = nn.Linear(z_dim, base_filters * 16 * self.n_prod)

        self.upconv4 = self._upconv(base_filters * 16, base_filters * 8)
        self.decoder4 = self._conv_block(base_filters * 16, base_filters * 8)
        self.upconv3 = self._upconv(base_filters * 8, base_filters * 4)
        self.decoder3 = self._conv_block(base_filters * 8, base_filters * 4)
        self.upconv2 = self._upconv(base_filters * 4, base_filters * 2)
        self.decoder2 = self._conv_block(base_filters * 4, base_filters * 2)
        self.upconv1 = self._upconv(base_filters * 2, base_filters)
        self.decoder1 = self._conv_block(base_filters * 2, base_filters)
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels, negative_slope=0.1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True)
        )
    

    def _upconv(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


    def forward(self, x):
        # Contracting path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool3d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool3d(2)(enc2))
        enc4 = self.encoder4(nn.MaxPool3d(2)(enc3))
        enc5 = self.encoder5(nn.MaxPool3d(2)(enc4))
        
        b = len(enc5)
        enc5 = self.mlp_1(enc5.view(b, -1))

        # Expanding path
        dec4 = self.upconv4(self.mlp_2(enc5).view(b, 128, *self.ns)) # HERE!
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)

        out[:,:x.shape[1]] = x

        return out, None



if __name__ == '__main__':

    model = Unet3D(in_channels=4, out_channels=9, base_filters=8)
    # init_unet_identityish(model, in_channels=4)
    init_unet_primary_skip(model)

    # Example usage
    batch_size = 1
    in_channels = 4
    depth, height, width = 80, 80, 48

    # Create a dummy input
    x = torch.randn(1, 4, 80, 80, 48)  # [batch=1, in_channels=4, D=80, H=80, W=48]
    y, _ = model(x)                    # y will be [1, 9, 80, 80, 48]

    # Compare first 4 channels of y to x
    mse = ((y[:, :4] - x)**2).mean().item()
    print(f"Initial MSE between input and first 4 output channels = {mse:.6f}")
