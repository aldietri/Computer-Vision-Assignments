import torch
import pytorch_lightning as pl
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import os
import math

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

# Init DataLoader from MNIST Dataset
# Init DataLoader from MNIST Dataset
batch_size = 512 

transform=transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))])
    
mnist_train = torchvision.datasets.MNIST('.', train=True, download=True, transform=transform)
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [50000, 10000])

mnist_test = torchvision.datasets.MNIST(os.getcwd(), train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, num_workers=8)
val_loader = torch.utils.data.DataLoader(mnist_val, batch_size=64, num_workers=8) 
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, num_workers=8)

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, model_dim, max_len, device):
        """
        constructor of sinusoid encoding class

        :param model_dim: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, model_dim, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, model_dim, step=2, device=device).float()
        # 'i' means index of model_dim (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / model_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / model_dim)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, model_dim = 512]

        batch_size, seq_len, N = x.size()
        # batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, model_dim = 512]
        # it will add with tok_emb : [128, 30, 512]         

class Attention(nn.Module):
    def __init__(self, hidden_dim, model_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        # TODO
        self.W_Q = nn.Linear(model_dim, hidden_dim, device=device)
        self.W_K = nn.Linear(model_dim, hidden_dim, device=device)
        self.W_V = nn.Linear(model_dim, hidden_dim, device=device)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # TODO

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        prob = self.softmax(scores)
        z = prob @ V

        return z      

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, model_dim, nr_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.nr_heads = nr_heads
        self.attention = []
        # TODO:
        self.W_concat = nn.Linear(nr_heads*hidden_dim, model_dim, device=device) 

        for _ in range(nr_heads):
            self.attention.append(Attention(hidden_dim, model_dim))

    def forward(self, x):
        # TODO:
        scores = []
        for attn_head in self.attention:
            scores.append(attn_head(x))

        z = torch.cat(scores, dim=-1)

        output = self.W_concat(z)

        return output

class LayerNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(model_dim)).to(device)
        self.beta = nn.Parameter(torch.zeros(model_dim)).to(device)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out

class FFN(nn.Module):
    def __init__(self, model_dim, hidden_dim, drop_prob=0.1):
        super(FFN, self).__init__()
        # TODO
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(drop_prob)
        )

    def forward(self, x):
        # TODO:
        return self.net(x)

class Transformer_module(nn.Module):
    def __init__(self, hidden_dim, model_dim, ffn_hidden, nr_heads):
        super(Transformer_module, self).__init__()
        # TODO:
        self.pos = PositionalEncoding(model_dim, 256**2, device) # Assumption that the max seq length >= 256**2, but can be an arbitrary number that should be sufficiently large
        self.multihead_attn = MultiHeadAttention(hidden_dim, model_dim, nr_heads)
        self.norm1 = LayerNorm(model_dim)
        self.norm2 = LayerNorm(model_dim)
        self.ffn = FFN(model_dim, ffn_hidden)

    def forward(self, x):
        # TODO:
        # 0. Positional encoding
        x += self.pos(x) 

        # 1. Norm
        x2 = self.norm1(x)
        
        # 2. Self attention
        x3 = self.multihead_attn(x2)

        # 3. Residual
        x4 = x3 + x

        # 4 Norm
        x5 = self.norm2(x4)

        # 5. MLP
        x6 = self.ffn(x5)

        # 6. Residual
        x7 = x6 + x4

        return x7

def split_up_patches(x, patch_size):
    h = x.shape[-2]
    w = x.shape[-1]
    patches = nn.Unfold(kernel_size = patch_size, stride = patch_size+1)(x)
    patches = torch.permute(patches, (0, 2, 1)) # note: index convention is (n_batches, n_tokens, hidden_dim)!
    return patches.to(device)

class Transformer(pl.LightningModule):
    def __init__(self, hidden_dim = 64, model_dim=128, ffn_hidden=256, num_class = 10, nr_layers = 3, nr_heads = 3, patch_size = 4):
        super().__init__()
        self.num_class = num_class
        self.nr_layers = nr_layers
        self.nr_heads = nr_heads
        self.learning_rate = 1e-3
        self.patch_size = patch_size

        self.train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes = num_class, average='weighted')
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(num_classes = num_class, average='weighted')
        self.test_acc = torchmetrics.classification.MulticlassAccuracy(num_classes = num_class, average='weighted')

        self.transformer_modules = []
        # TODO
        # Patch Embedding
        self.lin = nn.Linear(patch_size**2, model_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, model_dim))

        # Transformer blocks
        for _ in range(nr_layers):
            self.transformer_modules.append(Transformer_module(hidden_dim, model_dim, ffn_hidden, nr_heads))

        self.transformer_blocks = nn.Sequential(*self.transformer_modules)
          
        # classification head
        self.mlp = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_class)
        )

    def forward(self, x):
        # TODO:
        x = split_up_patches(x, self.patch_size)
        x = self.lin(x)

        cls_tokens = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer_blocks(x)
    
        x = self.mlp(x[:, 0])

        return x

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch

        # Forward pass
        outputs = self(images)
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(outputs, labels)
        self.log('train_loss', loss)

        pred_labels = torch.argmax(outputs, 1)
        self.train_acc(pred_labels, labels)

        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        outputs = self(images)
        pred_labels = torch.argmax(outputs, 1)

        self.val_acc(pred_labels, labels)

        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        pred_labels = torch.argmax(outputs, 1)

        self.test_acc(pred_labels, labels)

        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return opt


trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=50)
model = Transformer(128, 256, 512, 10, 8, 8, 4)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)