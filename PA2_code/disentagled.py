import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters remain the same
batch_size = 16
block_size = 32
learning_rate = 1e-3
model_dim = 64
n_head = 2
n_layer = 4
dropout = 0.2
n_output = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DisentangledHead(nn.Module):
    """Disentangled attention head implementing DeBERTa-style attention"""
    def __init__(self, head_size):
        super().__init__()
        # Content projections
        self.key_content = nn.Linear(model_dim, head_size, bias=False)
        self.query_content = nn.Linear(model_dim, head_size, bias=False)
        self.value = nn.Linear(model_dim, head_size, bias=False)
        
        # Position projections
        self.key_position = nn.Linear(model_dim, head_size, bias=False)
        self.query_position = nn.Linear(model_dim, head_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, content, position):
        B, T, C = content.shape
        
        # Content-to-content attention
        q_c = self.query_content(content)  # (B,T,head_size)
        k_c = self.key_content(content)    # (B,T,head_size)
        
        # Position-to-content attention
        q_p = self.query_position(position)  # (T,head_size)
        k_p = self.key_position(position)    # (T,head_size)
        
        # Calculate attention scores
        # Content-content attention
        cc_attention = (q_c @ k_c.transpose(-2, -1)) * (k_c.shape[-1] ** -0.5)  # (B,T,T)
        
        # Position-position attention
        pp_attention = (q_p @ k_p.transpose(-2, -1)) * (k_p.shape[-1] ** -0.5)  # (T,T)
        
        # Combine attention patterns
        attention = cc_attention + pp_attention.unsqueeze(0)  # Broadcasting for batch
        
        # Apply softmax BEFORE storing attention weights
        attention = F.softmax(attention, dim=-1)
        
        # Store normalized attention weights
        attention_weights = attention.clone()
        
        # Apply dropout after softmax and storing weights
        attention = self.dropout(attention)
        
        # Apply attention to values
        v = self.value(content)
        output = attention @ v
        
        return output, attention_weights

class DisentangledMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([DisentangledHead(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, content, position):
        outputs = []
        attention_weights = []
        
        for head in self.heads:
            head_output, head_weights = head(content, position)
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


class DisentangledEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        head_size = model_dim // num_heads
        self.attention = DisentangledMultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForwardLayer(model_dim)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, x, pos_emb):
        attn_out, attention_weights = self.attention(self.ln1(x), pos_emb)
        x = x + attn_out
        x = x + self.feed_forward(self.ln2(x))
        return x, attention_weights

class DistangledEncoderModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, model_dim)
        self.position_embedding_table = nn.Embedding(block_size, model_dim)
        self.blocks = nn.ModuleList([DisentangledEncoderBlock(model_dim, n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, n_output)
        
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
        
        # Get embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        x = tok_emb
        attention_maps = []
        
        for block in self.blocks:
            x, attn_weights = block(x, pos_emb)
            attention_maps.append(attn_weights)
        
        x = self.ln_final(x)
        x = x.mean(dim=1)  # Pool over sequence length
        # logits = self.classifier(x)
        
        # if targets is None:
        #     loss = None
        # else:
        #     loss = F.cross_entropy(logits, targets)
            
        return x, attention_maps

class DisentangledDecoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        head_size = model_dim // num_heads
        self.attention = DisentangledMultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForwardLayer(model_dim)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, pos_emb):
        B, T, C = x.shape
        
        # Apply causal mask to attention
        mask = self.mask[:T, :T]
        
        # Get attention output with masking
        attn_out, attention_weights = self.attention(self.ln1(x), pos_emb)
        attention_weights = attention_weights.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        x = x + attn_out
        x = x + self.feed_forward(self.ln2(x))
        return x, attention_weights

class DisentagledDecoderModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, model_dim)
        self.position_embedding_table = nn.Embedding(block_size, model_dim)
        self.blocks = nn.ModuleList([DisentangledDecoderBlock(model_dim, n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(model_dim)
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
        
        # Get embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        x = tok_emb
        attention_maps = []
        
        for block in self.blocks:
            x, attn_weights = block(x, pos_emb)
            attention_maps.append(attn_weights)
        
        x = self.ln_final(x)
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
        for _ in range(max_new_tokens):
            # Take last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get predictions
            logits, _, _ = self(idx_cond)
            # Focus on last time step
            logits = logits[:, -1, :]
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx