# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

from config import (
    ENCODER_TOP_CONFIG, QUANTIZER_TOP_CONFIG,
    ENCODER_BOTTOM_CONFIG, QUANTIZER_BOTTOM_CONFIG,
    DECODER_CONFIG
)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(), # CORRECTED: Was nn.ReLU(True)
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(), # CORRECTED: Was nn.ReLU(True)
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)
        )
    def forward(self, x):
        return x + self._block(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs_permuted.shape
        flat_input = inputs_permuted.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs_permuted)
        q_latent_loss = F.mse_loss(quantized, inputs_permuted.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs_permuted + (quantized - inputs_permuted).detach()
        return loss, quantized.permute(0, 3, 1, 2).contiguous()


class HierarchicalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_residual_layers, num_residual_hiddens, downsample_factor):
        super().__init__()
        
        if downsample_factor == 1:
            layers = [nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        elif downsample_factor == 0.5:
            layers = [
                nn.ConvTranspose2d(in_channels, hidden_channels//2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(), # CORRECTED: Was nn.ReLU(True)
                nn.Conv2d(hidden_channels//2, hidden_channels, kernel_size=3, padding=1)
            ]
        else:
            # This 'else' block handles downsampling by integer factors, which you are not currently using.
            # Correcting it for completeness.
            num_downsample_layers = int(torch.log2(torch.tensor(downsample_factor)).item())
            layers = [nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1), nn.ReLU()]
            for _ in range(num_downsample_layers - 1):
                layers.append(nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=4, stride=2, padding=1))
                layers.append(nn.ReLU())
            layers.append(nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1))
        
        self.main = nn.Sequential(*layers)
        
        self.res_stack = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels, num_residual_hiddens) 
            for _ in range(num_residual_layers)
        ])

    def forward(self, x):
        x = self.main(x)
        for layer in self.res_stack:
            x = layer(x)
        return x


# --- SUPER-RESOLUTION : A NEW CONDITIONAL DECODER ---
class ConditionalHierarchicalDecoder(nn.Module):
    def __init__(self, top_embedding_dim, bottom_embedding_dim, hidden_channels, num_residual_layers, num_residual_hiddens, out_channels):
        super().__init__()
        self.lr_condition_upsampler = nn.Sequential(
            nn.ConvTranspose2d(config.IMAGE_CHANNELS, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), # CORRECTED: Was nn.ReLU(True)
            ResidualBlock(in_channels=hidden_channels, num_hiddens=hidden_channels, num_residual_hiddens=num_residual_hiddens),
            nn.ReLU()  # CORRECTED: Was nn.ReLU(True)
        )
        
        self.pre_conv_top = nn.Conv2d(top_embedding_dim, hidden_channels, kernel_size=3, padding=1)
        self.pre_conv_bottom = nn.Conv2d(bottom_embedding_dim, hidden_channels, kernel_size=3, padding=1)
        
        self.upsample_top = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU() # CORRECTED: Was nn.ReLU(True)
        )
        self.fusion_conv = nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1)
        self.res_stack = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels, num_residual_hiddens) 
            for _ in range(num_residual_layers)
        ])
        
        # CORRECTED: Use a standard Conv2d because the feature map is already at the target resolution.
        self.upsample_final = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z_top, z_bottom, lr_condition):
        z_top_proc = self.pre_conv_top(z_top)    
        z_top_upsampled = self.upsample_top(z_top_proc) 
        z_bottom_proc = self.pre_conv_bottom(z_bottom)  
        z_bottom_upsampled = self.upsample_top(z_bottom_proc)
        lr_features = self.lr_condition_upsampler(lr_condition) 
        x = torch.cat([z_top_upsampled, z_bottom_upsampled, lr_features], dim=1)  
        x = self.fusion_conv(x)  
        
        for layer in self.res_stack:
            x = layer(x)
        x = self.upsample_final(x)  
        return x


# --- SUPER-RESOLUTION : The Main VQ-VAE-2 Model ---
class VQVAE2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_top = HierarchicalEncoder(**ENCODER_TOP_CONFIG)
        self.quantizer_top = VectorQuantizer(**QUANTIZER_TOP_CONFIG)
        self.pre_quant_conv_top = nn.Conv2d(ENCODER_TOP_CONFIG['hidden_channels'], QUANTIZER_TOP_CONFIG['embedding_dim'], kernel_size=1)
        
        self.encoder_bottom = HierarchicalEncoder(**ENCODER_BOTTOM_CONFIG)
        self.quantizer_bottom = VectorQuantizer(**QUANTIZER_BOTTOM_CONFIG)
        self.pre_quant_conv_bottom = nn.Conv2d(ENCODER_BOTTOM_CONFIG['hidden_channels'], QUANTIZER_BOTTOM_CONFIG['embedding_dim'], kernel_size=1)
        self.decoder = ConditionalHierarchicalDecoder(**DECODER_CONFIG)

    def forward(self, lr_image):
        z_top_unquantized = self.encoder_top(lr_image)
        z_top_unquantized = self.pre_quant_conv_top(z_top_unquantized)
        vq_loss_top, z_top_quantized = self.quantizer_top(z_top_unquantized)
        
        z_bottom_unquantized = self.encoder_bottom(lr_image)
        z_bottom_unquantized = self.pre_quant_conv_bottom(z_bottom_unquantized)
        vq_loss_bottom, z_bottom_quantized = self.quantizer_bottom(z_bottom_unquantized)
        reconstructed_image = self.decoder(z_top_quantized, z_bottom_quantized, lr_image)
        total_vq_loss = vq_loss_top + vq_loss_bottom
        return reconstructed_image, total_vq_loss

    def get_shared_state_dict(self):
    # Aggregate EVERYTHING (encoder + quantizers + decoder)
        return {k: v.clone().detach().cpu() for k, v in self.state_dict().items()}

    def load_shared_state_dict(self, state_dict):
        # Strict to ensure every param is synchronized
        self.load_state_dict(state_dict, strict=True)
