# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
from torch.nn import functional as F

# The list of my Hyperparameters
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
class Head(nn.Module):
    """ My head for self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(model_dim, head_size, bias=False)
        self.query = nn.Linear(model_dim, head_size, bias=False)
        self.value = nn.Linear(model_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Batch Size, Sequence Length, Embedding Dimension
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head size)
        q = self.query(x) # (B,T,head size)
        # computing attention scores using the scaled dot product
        weight = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        
        # Store the attention weights BEFORE softmax and dropout
        raw_weights = weight.clone()
        
        weight = F.softmax(weight, dim=-1) # (B, T, T)
        
        # Store normalized weights BEFORE dropout
        attention_weights = weight.clone()
        
        # Now apply dropout
        weight = self.dropout(weight)
        
        # performing the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        output = weight @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
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

class Block(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        head_size = model_dim//num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffw = FeedForwardLayer(model_dim)
        self.l1 = nn.LayerNorm(model_dim)
        self.l2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        sa_out, attention_weights = self.sa(self.l1(x))
        x = x + sa_out
        x = x + self.ffw(self.l2(x))
        return x, attention_weights

class EncoderModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, model_dim)
        self.position_embedding_table = nn.Embedding(block_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, n_head) for _ in range(n_layer)])
        self.lf = nn.LayerNorm(model_dim)
        # self.classifier = nn.Sequential(
        #     nn.Linear(model_dim, n_output)
        # )
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
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)
        
        x = self.lf(x)
        #logits = self.lm_head(x)
        x = x.mean(dim=1)
        return x, attention_maps

class MaskedHead(nn.Module):
    """ My head for self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(model_dim, head_size, bias=False)
        self.query = nn.Linear(model_dim, head_size, bias=False)
        self.value = nn.Linear(model_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Batch Size, Sequence Length, Embedding Dimension
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head size)
        q = self.query(x) # (B,T,head size)
        # computing attention scores using the scaled dot product
        weight = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weight = F.softmax(weight, dim=-1) # (B, T, T) now applying softmax so e power -inf becomes zero
        
        # Store normalized weights BEFORE dropout
        attention_weights = weight.clone()
        
        weight = self.dropout(weight)
        # performing the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        output = weight @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return output, attention_weights

class MultiHeadMaskedAttention(nn.Module):
    """
        My multiple heads in the self attention mechanism 
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MaskedHead(head_size) for i in range(num_heads)])
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
        # output = torch.cat([h(x) for h in self.heads], dim=-1)
        # output = self.dropout(self.projection(output)) # Linear projection for the output layer
        # return output

class BlockDecoder(nn.Module):
    """
        My transformer block to add the attention layers and all the feedforward layers
    """
    def __init__(self, model_dim, num_heads):
        super().__init__()
        head_size = model_dim//num_heads
        self.sa = MultiHeadMaskedAttention(num_heads, head_size)
        self.ffw = FeedForwardLayer(model_dim)
        self.l1 = nn.LayerNorm(model_dim)
        self.l2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        sa_out, attention_weights = self.sa(self.l1(x))
        x = x + sa_out
        x = x + self.ffw(self.l2(x))
        return x, attention_weights
        # x = x + self.sa(self.l1(x))
        # x = x + self.ffw(self.l2(x))
        # #print(x[:5])    
        # return x 

class DecoderModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, model_dim)
        self.position_embedding_table = nn.Embedding(block_size, model_dim)
        self.blocks = nn.Sequential(*[BlockDecoder(model_dim, n_head) for _ in range(n_layer)])
        self.lf = nn.LayerNorm(model_dim) # the final layer normalization
        # self.classifier = nn.Sequential(
        #     nn.Linear(model_dim, n_output)  # Direct mapping to output classes
        # )
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

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        #x = self.blocks(x) # (B,T,C)

        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)
        
        x = self.lf(x)
        

        # Use mean pooling instead of just the last token
        #x = x.mean(dim=1)  # (B,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
         # Classification layer
        #logits = self.classifier(x)  # (B,num_classes)

        #print(logits)
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
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

