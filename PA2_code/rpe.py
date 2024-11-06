import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (keeping same as original)
batch_size = 16
block_size = 32
learning_rate = 1e-3
n_embd = 64
model_dim = 64
n_head = 2
n_layer = 4
num_heads = 2
dropout = 0.2
n_output = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RelativePositionEncoding(nn.Module):
    def __init__(self, max_relative_position, head_size):
        super().__init__()
        self.max_relative_position = max_relative_position
        # Create relative position embeddings for keys and values
        self.relative_key_embedding = nn.Embedding(
            2 * max_relative_position + 1, head_size)
        self.relative_value_embedding = nn.Embedding(
            2 * max_relative_position + 1, head_size)
    
    def _relative_position_bucket(self, relative_position):
        ret = relative_position.clone()
        ret = torch.clamp(ret, -self.max_relative_position, self.max_relative_position)
        ret += self.max_relative_position  # shift to positive indices
        return ret

    def forward(self, length):
        # Create relative position matrix
        range_vec = torch.arange(length, device=device)
        range_mat = range_vec.unsqueeze(0).expand(length, -1)
        relative_position = range_mat - range_mat.transpose(0, 1)
        
        relative_position_bucket = self._relative_position_bucket(relative_position)
        
        rel_key_embeddings = self.relative_key_embedding(relative_position_bucket)
        rel_value_embeddings = self.relative_value_embedding(relative_position_bucket)
        
        return rel_key_embeddings, rel_value_embeddings

class RelativeAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(model_dim, head_size, bias=False)
        self.query = nn.Linear(model_dim, head_size, bias=False)
        self.value = nn.Linear(model_dim, head_size, bias=False)
        self.rel_pos = RelativePositionEncoding(
            max_relative_position=block_size//2,
            head_size=head_size
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Regular transformations
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        # Get relative position embeddings
        rel_k, rel_v = self.rel_pos(T)  # (T, T, head_size)
        
        # Reshape query for batch-wise matrix multiplication with rel_k
        # From (B, T, head_size) to (B, T, 1, head_size)
        q_reshaped = q.unsqueeze(2)
        
        # Reshape rel_k to work with batched queries
        # From (T, T, head_size) to (1, T, T, head_size)
        rel_k = rel_k.unsqueeze(0)
        rel_v = rel_v.unsqueeze(0)
        
        # Compute attention scores
        content_scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)  # (B, T, T)
        
        # Compute position scores with correct broadcasting
        # q_reshaped: (B, T, 1, head_size)
        # rel_k: (1, T, T, head_size)
        # Result after transpose and matmul: (B, T, T)
        position_scores = torch.matmul(q_reshaped, rel_k.transpose(-2, -1)).squeeze(2) * (self.head_size ** -0.5)
        
        attention_scores = content_scores + position_scores
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        wei = attention_weights.clone()
        attention_weights = self.dropout(attention_weights)
        
        # Compute output
        content_output = attention_weights @ v  # (B, T, head_size)
        
        # Compute position-weighted values
        # attention_weights: (B, T, T)
        # rel_v: (1, T, T, head_size)
        position_output = torch.matmul(attention_weights.unsqueeze(2), rel_v).squeeze(2)
        
        output = content_output + position_output
        
        return output, wei
    
class MultiHeadRelativeAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([RelativeAttentionHead(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        outputs = []
        attention_weights = []
        
        for head in self.heads:
            head_output, head_weights = head(x, mask)
            outputs.append(head_output)
            attention_weights.append(head_weights)
        
        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.projection(out))
        return out, torch.stack(attention_weights)

class EncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        head_size = model_dim // num_heads
        self.attention = MultiHeadRelativeAttention(num_heads, head_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        attended, attention_weights = self.attention(self.ln1(x))
        x = x + attended
        x = x + self.feed_forward(self.ln2(x))
        return x, attention_weights

class RelativePositionEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([EncoderBlock(model_dim, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(model_dim)
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
        
        x = self.token_embedding(idx)
        
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)
        
        x = self.ln_f(x)
        x = x.mean(dim=1)  # pool over sequence
        return x, attention_maps

class DecoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        head_size = model_dim // num_heads
        self.attention = MultiHeadRelativeAttention(num_heads, head_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        # Create causal mask
        size = x.size(1)
        causal_mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        causal_mask = ~causal_mask
        
        attended, attention_weights = self.attention(self.ln1(x), mask=causal_mask)
        x = x + attended
        x = x + self.feed_forward(self.ln2(x))
        return x, attention_weights

class RelativePositionDecoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([DecoderBlock(model_dim, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(model_dim)
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
        
        x = self.token_embedding(idx)
        
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)
        
        x = self.ln_f(x)
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
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx