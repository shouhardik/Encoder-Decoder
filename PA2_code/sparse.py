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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LocalWindowHead(nn.Module):
    """Non-causal attention head with local window attention pattern"""
    
    def __init__(self, head_size, window_size=8):
        super().__init__()
        self.key = nn.Linear(model_dim, head_size, bias=False)
        self.query = nn.Linear(model_dim, head_size, bias=False)
        self.value = nn.Linear(model_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        
        # Create sliding window attention mask
        # For encoder, we allow attention to both previous and future tokens within window
        mask = torch.ones(block_size, block_size, dtype=torch.bool)
        for i in range(block_size):
            # Calculate window boundaries
            window_start = max(0, i - window_size // 2)
            window_end = min(block_size, i + window_size // 2 + 1)
            mask[i, window_start:window_end] = False
            
        self.register_buffer("attention_mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        v = self.value(x)  # (B,T,head_size)

        # Compute attention scores
        weight = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B,T,T)
        
        # Apply local window mask
        # Use only the relevant part of the mask based on sequence length
        mask = self.attention_mask[:T, :T]
        #weight = weight.masked_fill(mask, float('-inf'))
        
        # Store attention weights before dropout
        attention_weights = F.softmax(weight, dim=-1)
        
        # Apply dropout
        weight = self.dropout(attention_weights)
        
        # Weighted aggregation of values
        output = weight @ v
        
        return output, attention_weights

class MultiHeadLocalWindowAttentionEncoder(nn.Module):
    """Multi-head attention with local window attention pattern for encoder"""
    
    def __init__(self, num_heads, head_size, window_size=8):
        super().__init__()
        self.heads = nn.ModuleList([
            LocalWindowHead(head_size, window_size) 
            for _ in range(num_heads)
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


class BlockSparseEncoder(nn.Module):
    """Transformer block with sparse local window attention for encoder"""
    
    def __init__(self, model_dim, num_heads, window_size=8):
        super().__init__()
        head_size = model_dim // num_heads
        self.sa = MultiHeadLocalWindowAttentionEncoder(num_heads, head_size, window_size)
        self.ffw = FeedForwardLayer(model_dim)
        self.l1 = nn.LayerNorm(model_dim)
        self.l2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        sa_out, attention_weights = self.sa(self.l1(x))
        x = x + sa_out
        x = x + self.ffw(self.l2(x))
        return x, attention_weights

class SparseEncoderModel(nn.Module):
    """Encoder model with sparse local window attention"""
    
    def __init__(self, vocab_size, window_size=8):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, model_dim)
        self.position_embedding_table = nn.Embedding(block_size, model_dim)
        self.blocks = nn.ModuleList([
            BlockSparseEncoder(model_dim, n_head, window_size) 
            for _ in range(n_layer)
        ])
        self.lf = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, n_output)
        )
        self.window_size = window_size
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
        x = tok_emb + pos_emb
        
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)
        
        x = self.lf(x)
        # Mean pooling instead of taking just the last token
        x = x.mean(dim=1)
        logits = self.classifier(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss, attention_maps

    def add_global_tokens(self):
        """
        Optional enhancement: Add global tokens that can attend to all positions
        This can be used to capture long-range dependencies while maintaining efficiency
        """
        class GlobalLocalWindowHead(LocalWindowHead):
            def __init__(self, head_size, window_size=8, num_global_tokens=2):
                super().__init__(head_size, window_size)
                self.num_global_tokens = num_global_tokens
                
                # Modify attention mask to allow global tokens
                mask = self.attention_mask.clone()
                # Global tokens can attend to all positions
                mask[:num_global_tokens, :] = False
                # All tokens can attend to global tokens
                mask[:, :num_global_tokens] = False
                self.register_buffer("attention_mask", mask)
                
        # Replace standard attention heads with global-local heads
        for block in self.blocks:
            head_size = model_dim // n_head
            block.sa = MultiHeadLocalWindowAttentionEncoder(
                n_head, 
                head_size, 
                self.window_size
            )
            block.sa.heads = nn.ModuleList([
                GlobalLocalWindowHead(head_size, self.window_size) 
                for _ in range(n_head)
            ])


class LocalWindowMaskedHead(nn.Module):
    """Masked attention head with local window attention pattern"""
    
    def __init__(self, head_size, window_size=8):
        super().__init__()
        self.key = nn.Linear(model_dim, head_size, bias=False)
        self.query = nn.Linear(model_dim, head_size, bias=False)
        self.value = nn.Linear(model_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        
        # Create sliding window attention mask
        mask = torch.ones(block_size, block_size, dtype=torch.bool)
        for i in range(block_size):
            # Allow attention to window_size previous tokens
            start_idx = max(0, i - window_size + 1)
            mask[i, start_idx:i+1] = False
            
        # Combine with causal mask to ensure no attention to future tokens
        causal_mask = ~torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        mask = mask | causal_mask
        
        self.register_buffer("attention_mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        v = self.value(x)  # (B,T,head_size)

        # Compute attention scores
        weight = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B,T,T)
        
        # Apply local window + causal mask
        # Use only the relevant part of the mask based on sequence length
        mask = self.attention_mask[:T, :T]
        weight = weight.masked_fill(mask, float('-inf'))
        
        # Store attention weights before dropout
        attention_weights = F.softmax(weight, dim=-1)
        
        # Apply dropout
        weight = self.dropout(attention_weights)
        
        # Weighted aggregation of values
        output = weight @ v
        
        return output, attention_weights

class MultiHeadLocalWindowAttention(nn.Module):
    """Multi-head attention with local window attention pattern"""
    
    def __init__(self, num_heads, head_size, window_size=8):
        super().__init__()
        self.heads = nn.ModuleList([
            LocalWindowMaskedHead(head_size, window_size) 
            for _ in range(num_heads)
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

class BlockSparseDecoder(nn.Module):
    """Transformer block with sparse local window attention"""
    
    def __init__(self, model_dim, num_heads, window_size=8):
        super().__init__()
        head_size = model_dim // num_heads
        self.sa = MultiHeadLocalWindowAttention(num_heads, head_size, window_size)
        self.ffw = FeedForwardLayer(model_dim)
        self.l1 = nn.LayerNorm(model_dim)
        self.l2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        sa_out, attention_weights = self.sa(self.l1(x))
        x = x + sa_out
        x = x + self.ffw(self.l2(x))
        return x, attention_weights

class SparseDecoderModel(nn.Module):
    """Decoder model with sparse local window attention"""
    
    def __init__(self, vocab_size, window_size=8):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, model_dim)
        self.position_embedding_table = nn.Embedding(block_size, model_dim)
        self.blocks = nn.ModuleList([
            BlockSparseDecoder(model_dim, n_head, window_size) 
            for _ in range(n_layer)
        ])
        self.lf = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size)
        self.window_size = window_size
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
        x = tok_emb + pos_emb
        
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