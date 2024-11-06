import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
model_dim = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers
num_heads = 2
dropout=0.2
n_output = 3  # Output size for the classifier, we have 3 classes

class HeadALiBi(nn.Module):
    """Head for self-attention with ALiBi positional encoding"""
    
    def __init__(self, head_size, head_index, num_heads):
        super().__init__()
        self.key = nn.Linear(model_dim, head_size, bias=False)
        self.query = nn.Linear(model_dim, head_size, bias=False)
        self.value = nn.Linear(model_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # ALiBi slope computation
        m = 2 ** -(8 / num_heads) * (head_index + 1)
        self.register_buffer(
            "alibi_bias",
            -m * torch.arange(block_size).unsqueeze(0).expand(block_size, -1)
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)

        # Compute attention scores
        weight = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B,T,T)
        
        # Add ALiBi positional bias
        # Use only the relevant part of the bias based on sequence length
        weight = weight + self.alibi_bias[:T, :T]
        
        # Store raw weights before softmax and dropout
        attention_weights = F.softmax(weight, dim=-1)
        
        # Apply dropout
        weight = self.dropout(attention_weights)
        
        # Weighted aggregation of values
        output = weight @ v
        
        return output, attention_weights

class MultiHeadAttentionALiBi(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([
            HeadALiBi(head_size, i, num_heads) 
            for i in range(num_heads)
        ])
        self.projection = nn.Linear(head_size * num_heads, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = []
        attention_weights = []
        for head in self.heads:
            head_output, head_weights = head(x)
            outputs.append(head_output)
            attention_weights.append(head_weights)
            
        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.projection(out))
        return out, torch.stack(attention_weights)
    
class FeedForwardLayer(nn.Module):
    """
        My Feedforward layer adding linear layers and dropout
    """
    def __init__(self, model_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, 4*model_dim),
            nn.ReLU(),
            nn.Linear(4*model_dim, model_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.net(x)

class BlockALiBi(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        head_size = model_dim // num_heads
        self.sa = MultiHeadAttentionALiBi(num_heads, head_size)
        self.ffw = FeedForwardLayer(model_dim)
        self.l1 = nn.LayerNorm(model_dim)
        self.l2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        sa_out, attention_weights = self.sa(self.l1(x))
        x = x + sa_out
        x = x + self.ffw(self.l2(x))
        return x, attention_weights

class EncoderModelALiBi(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Remove position embedding table since we use ALiBi
        self.token_embedding_table = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([BlockALiBi(model_dim, n_head) for _ in range(n_layer)])
        self.lf = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, n_output)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Only use token embeddings, no position embeddings needed
        x = self.token_embedding_table(idx)
        
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)
        
        x = self.lf(x)
        x = x.mean(dim=1)
            
        return x, attention_maps
    
class MaskedHeadALiBi(nn.Module):
    """Masked Head for self-attention with ALiBi positional encoding"""
    
    def __init__(self, head_size, head_index, num_heads):
        super().__init__()
        self.key = nn.Linear(model_dim, head_size, bias=False)
        self.query = nn.Linear(model_dim, head_size, bias=False)
        self.value = nn.Linear(model_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # ALiBi slope computation
        m = 2 ** -(8 / num_heads) * (head_index + 1)
        self.register_buffer(
            "alibi_bias",
            -m * torch.arange(block_size).unsqueeze(0).expand(block_size, -1)
        )
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)

        # Compute attention scores
        weight = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B,T,T)
        
        # Add causal mask
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Add ALiBi positional bias
        # Use only the relevant part of the bias based on sequence length
        weight = weight + self.alibi_bias[:T, :T]
        
        # Store weights before dropout for visualization
        attention_weights = F.softmax(weight, dim=-1)
        
        # Apply dropout
        weight = self.dropout(attention_weights)
        
        # Weighted aggregation of values
        output = weight @ v
        
        return output, attention_weights

class MultiHeadMaskedAttentionALiBi(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([
            MaskedHeadALiBi(head_size, i, num_heads) 
            for i in range(num_heads)
        ])
        self.projection = nn.Linear(head_size * num_heads, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = []
        attention_weights = []
        for head in self.heads:
            head_output, head_weights = head(x)
            outputs.append(head_output)
            attention_weights.append(head_weights)
            
        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.projection(out))
        return out, torch.stack(attention_weights)

class BlockDecoderALiBi(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        head_size = model_dim // num_heads
        self.sa = MultiHeadMaskedAttentionALiBi(num_heads, head_size)
        self.ffw = FeedForwardLayer(model_dim)
        self.l1 = nn.LayerNorm(model_dim)
        self.l2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        sa_out, attention_weights = self.sa(self.l1(x))
        x = x + sa_out
        x = x + self.ffw(self.l2(x))
        return x, attention_weights

class DecoderModelALiBi(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Remove position embedding table since we use ALiBi
        self.token_embedding_table = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([BlockDecoderALiBi(model_dim, n_head) for _ in range(n_layer)])
        self.lf = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Only use token embeddings, no position embeddings needed with ALiBi
        x = self.token_embedding_table(idx)
        
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)
        
        x = self.lf(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss, attention_maps

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx