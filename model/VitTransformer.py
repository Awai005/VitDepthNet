import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt



class Patches_loop_forward(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch_size, channels, height, width = images.size()
        pad_height = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_height > 0 or pad_width > 0:
            images = F.pad(images, (0, pad_width, 0, pad_height), mode='constant', value=0)
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size * self.patch_size)
        return patches.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, channels * self.patch_size * self.patch_size)

class PatchEncoder_loop_forward(nn.Module):
    def __init__(self, projection_dim, max_patches=1000):
        super().__init__()
        self.projection_dim = projection_dim
        self.max_patches = max_patches
        # Initialize with a large enough size to handle expected input, but consider making it dynamic if sizes can vary significantly
        self.position_embeddings = nn.Parameter(torch.randn(1, max_patches, projection_dim)).cuda()
        self.projection = None

    def forward(self, x, patch_dim):
        if self.projection is None or self.projection.in_features != patch_dim:
            self.projection = nn.Linear(patch_dim, self.projection_dim).cuda()

        # Adjust position_embeddings if the current number of patches exceeds the size
        current_patches = x.size(1)
        if current_patches > self.position_embeddings.size(1):
            # Extend position_embeddings to handle more patches
            new_embeddings = torch.randn(1, current_patches, self.projection_dim, device=self.position_embeddings.device)
            new_embeddings[:, :self.position_embeddings.size(1), :] = self.position_embeddings
            self.position_embeddings = nn.Parameter(new_embeddings)
            self.max_patches = current_patches

        x = self.projection(x)
        return x + self.position_embeddings[:, :current_patches, :]

class TransformerEncoderLayer_loop_forward(nn.Module):
    def __init__(self, projection_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(projection_dim).cuda()
        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout=dropout_rate).cuda()
        self.norm2 = nn.LayerNorm(projection_dim).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim).cuda(),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, projection_dim).cuda(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, src):
        src2 = self.norm1(src)
        src = src + self.attn(src2, src2, src2)[0]
        src2 = self.norm2(src)
        src = src + self.mlp(src2)
        return src

class VisionTransformerDepthEstimator_loop_forward(nn.Module):
    def __init__(self, patch_size=4, projection_dim=256, num_heads=8, transformer_layers=8, max_patches=1000):
        super().__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.patches = Patches_loop_forward(patch_size).cuda()
        self.encoder = PatchEncoder_loop_forward(projection_dim, max_patches).cuda()
        self.transformers = nn.Sequential(*[
            TransformerEncoderLayer_loop_forward(projection_dim, num_heads, 2 * projection_dim, 0.1).cuda() for _ in range(transformer_layers)
        ])
        self.to_feature_map = None

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = self.patches(x)
        num_patches = x.size(1)
        patch_dim = channels * self.patch_size * self.patch_size
        x = self.encoder(x, patch_dim)
        x = self.transformers(x)

        output_features = int((channels * height * width) / num_patches)
        if self.to_feature_map is None or self.to_feature_map.out_features != output_features:
            self.to_feature_map = nn.Linear(self.projection_dim, output_features).cuda()

        x = self.to_feature_map(x)
        return x.view(batch_size, channels, height, width)




class Patches_loop(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch_size, channels, height, width = images.size()
        pad_height = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_height > 0 or pad_width > 0:
            images = F.pad(images, (0, pad_width, 0, pad_height), mode='constant', value=0)
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size * self.patch_size)
        return patches.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, channels * self.patch_size * self.patch_size)

class PatchEncoder_loop(nn.Module):
    def __init__(self, projection_dim, max_patches):
        super().__init__()
        self.projection_dim = projection_dim
        self.max_patches = max_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, max_patches, projection_dim)).cuda()
        self.projection = None

    def forward(self, x, patch_dim):
        if self.projection is None or self.projection.in_features != patch_dim:
            self.projection = nn.Linear(patch_dim, self.projection_dim).cuda()
        x = self.projection(x)
        return x + self.position_embeddings[:, :x.size(1), :]

class TransformerEncoderLayer_loop(nn.Module):
    def __init__(self, projection_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(projection_dim).cuda()
        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout=dropout_rate).cuda()
        self.norm2 = nn.LayerNorm(projection_dim).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim).cuda(),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, projection_dim).cuda(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, src):
        src2 = self.norm1(src)
        src = src + self.attn(src2, src2, src2)[0]
        src2 = self.norm2(src)
        src = src + self.mlp(src2)
        return src

class VisionTransformerDepthEstimator_loop(nn.Module):
    def __init__(self, patch_size=4, projection_dim=128, num_heads=4, transformer_layers=4, max_patches=1000):
        super().__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.patches = Patches_loop(patch_size).cuda()
        self.encoder = PatchEncoder_loop(projection_dim, max_patches).cuda()
        self.transformers = nn.Sequential(*[
            TransformerEncoderLayer_loop(projection_dim, num_heads, 2 * projection_dim, 0.1).cuda() for _ in range(transformer_layers)
        ])
        self.to_feature_map = None

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = self.patches(x)
        num_patches = x.size(1)
        patch_dim = channels * self.patch_size * self.patch_size
        x = self.encoder(x, patch_dim)
        x = self.transformers(x)

        output_features = int((channels * height * width) / num_patches)
        if self.to_feature_map is None or self.to_feature_map.out_features != output_features:
            self.to_feature_map = nn.Linear(self.projection_dim, output_features).cuda()

        x = self.to_feature_map(x)
        return x.view(batch_size, channels, height, width)



# Vision transformer before or after bottleneck


class Patches_bn(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch_size, channels, height, width = images.size()
        # Handling cases where the dimensions are not divisible by the patch size
        pad_height = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - width % self.patch_size) % self.patch_size
        images = F.pad(images, (0, pad_width, 0, pad_height), 'constant', 0)
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, channels * self.patch_size * self.patch_size)
        return patches

class PatchEncoder_bn(nn.Module):
    def __init__(self, projection_dim):
        super().__init__()
        self.projection_dim = projection_dim
        self.projection = None
        self.position_embeddings = None

    def forward(self, x, num_patches, patch_dim):
        # Initialize the linear projection layer if it has not been initialized
        # or if the patch dimension size has changed.
        if self.projection is None or self.projection.in_features != patch_dim:
            self.projection = nn.Linear(patch_dim, self.projection_dim).cuda()

        # Initialize or update the position embeddings based on the current number of patches
        if self.position_embeddings is None or self.position_embeddings.size(1) < num_patches:
            # We resize to the new required size which is the maximum encountered num_patches
            self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, self.projection_dim)).cuda()

        x = self.projection(x)
        # Apply position embeddings to the projected data, adjusting to the current batch's num_patches
        return x + self.position_embeddings[:, :x.size(1), :]

class TransformerEncoderLayer_bn(nn.Module):
    def __init__(self, projection_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(projection_dim).cuda()
        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout=dropout_rate).cuda()
        self.norm2 = nn.LayerNorm(projection_dim).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim).cuda(),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, projection_dim).cuda(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, src):
        src2 = self.norm1(src)
        src = src + self.attn(src2, src2, src2)[0]
        src2 = self.norm2(src)
        src = src + self.mlp(src2)
        return src

class VisionTransformerDepthEstimator_bn(nn.Module):
    def __init__(self, patch_size=4, projection_dim=256, num_heads=16, transformer_layers=16, save_path='feature_maps'):
        super().__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim

        self.patches = Patches_bn(patch_size).cuda()
        self.encoder = PatchEncoder_bn(projection_dim).cuda()
        self.transformers = nn.Sequential(*[
            TransformerEncoderLayer_bn(projection_dim, num_heads, 2 * projection_dim, 0.1).cuda() for _ in range(transformer_layers)
        ])
        #self.to_feature_map = None  # This will be dynamically initialized
        #self.to_feature_map = nn.Linear(projection_dim, 16 * 2 * 128).cuda()
        self.to_feature_map = nn.Linear(projection_dim, 256 * 16).cuda() 
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        #self.save_tensor_as_image(x[0], 'initial_input.png')


        x = self.patches(x)
        num_patches = x.size(1)
        patch_dim = channels * self.patch_size * self.patch_size
        x = self.encoder(x, num_patches, patch_dim)
        x = self.transformers(x)
        
        self.to_feature_map = nn.Linear(self.projection_dim, 256 * 16).cuda()
        
        if width != height:
            self.to_feature_map = nn.Linear(self.projection_dim, 256 * 15).cuda()
        
        x = self.to_feature_map(x)
        
        x = x.view(batch_size, channels, height, width)  
        
        #self.save_tensor_as_image(x[0], 'reshaped_output.png')
        
        return x
    
    def save_tensor_as_image(self, tensor, filename):
        tensor = tensor.cpu().detach()
        # Assuming tensor is in the format (channels, height, width), select the first channel
        if tensor.dim() == 3:
            tensor = tensor[0]  # Select the first channel
        elif tensor.dim() == 4:
            tensor = tensor[0, 0]  # Select the first channel of the first batch item if batched
        plt.figure()
        plt.imshow(tensor, cmap='gray', interpolation='nearest')  # Use a grayscale colormap
        plt.axis('off')
        plt.savefig(os.path.join(self.save_path, filename), bbox_inches='tight', pad_inches=0)
        plt.close()
        
    # grayscale
    #def save_tensor_as_image(self, tensor, filename):
    #    tensor = tensor.cpu().detach()
    #    if tensor.shape[2] > 3:
    #        # Assume the tensor shape is (H, W, C), reduce to (H, W) by taking mean across channels for visualization
    #        tensor = tensor.mean(dim=2)
    #    plt.figure()
    #    plt.imshow(tensor, cmap='gray', interpolation='nearest')  # Use a grayscale colormap
    #    plt.axis('off')
    #    plt.savefig(os.path.join(self.save_path, filename), bbox_inches='tight', pad_inches=0)
    #    plt.close()



#VIT Transformer before conv_out
class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch_size, channels, height, width = images.size()
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, channels * self.patch_size * self.patch_size)
        #print("Patches shape:", patches.shape)
        return patches

class PatchEncoder(nn.Module):
    def __init__(self, patch_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(patch_dim, projection_dim).cuda()
        self.position_embeddings = nn.Parameter(torch.randn(1, 64, projection_dim)).cuda()  # Correct shape

    def forward(self, x):
        #print("Input to PatchEncoder:", x.shape)
        x = self.projection(x)
        x += self.position_embeddings
        #print("Output from PatchEncoder:", x.shape)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, projection_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(projection_dim).cuda()
        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout=dropout_rate).cuda()
        self.norm2 = nn.LayerNorm(projection_dim).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim).cuda(),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, projection_dim).cuda(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, src):
        src2 = self.norm1(src)
        src = src + self.attn(src2, src2, src2)[0]
        src2 = self.norm2(src)
        src = src + self.mlp(src2)
        return src

class VisionTransformerDepthEstimator(nn.Module):
    def __init__(self, patch_size=16, projection_dim=256, num_heads=8, transformer_layers=8):
        super().__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim

        self.patches = Patches(patch_size).cuda()
        self.encoder = None
        self.transformers = nn.Sequential(*[
            TransformerEncoderLayer(projection_dim, num_heads, 2 * projection_dim, 0.1).cuda() for _ in range(transformer_layers)
        ])
        self.to_feature_map = nn.Linear(projection_dim, 16 * 2 * 128).cuda()
        

    def forward(self, x):
        self.channels = x.size(1)
        num_patches = (x.size(-2) // self.patch_size) * (x.size(-1) // self.patch_size)
        patch_dim = self.channels * self.patch_size * self.patch_size
        self.encoder = PatchEncoder(patch_dim, self.projection_dim).cuda()

        x = self.patches(x)
        x = self.encoder(x)
        x = self.transformers(x)
        #print("Output from transformer:", x.shape)
        x = self.to_feature_map(x)
        #print("Output from feature_MAP:", x.shape)
        x = x.view(-1, 16, 128, 128)
        return x