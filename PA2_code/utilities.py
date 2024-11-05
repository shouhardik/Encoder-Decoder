
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from typing import List, Optional
from tokenizer import SimpleTokenizer
import os

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        _, _, attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        print(attn_maps[:5])

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0], dim=-1)
            #total_prob_over_rows = np.sum(att_map, axis=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()  
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map {j + 1}")
            
            # Save the plot
            plt.savefig(f"attention_map_{j + 1}.png")
            
            # Show the plot
            plt.show()
    def sanity_check01(self, sentence, block_size):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)
        
        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)
        
        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)
        
        # Process the input tensor through the encoder model
        _, _, attention_maps = self.model(input_tensor)
        
        # Display the number of attention maps
        print("Number of layers with attention maps:", len(attention_maps))
        
        # Visualize and save the attention maps
        for layer_idx, layer_attention in enumerate(attention_maps):
            # layer_attention shape is [num_heads, batch_size, seq_len, seq_len]
            num_heads = layer_attention.shape[0]
            
            for head_idx in range(num_heads):
                # Extract attention weights for this head
                # Remove batch dimension and convert to tensor
                att_map = layer_attention[head_idx, 0].detach().cpu()
                
                # Important: Check if the attention probabilities sum to 1 over rows
                total_prob_over_rows = torch.sum(att_map, dim=-1)  # Sum over the key dimension
                
                if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                    print(f"Layer {layer_idx}, Head {head_idx}")
                    print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                    print("Total probability over rows:", total_prob_over_rows.numpy())
                    print("Min sum:", total_prob_over_rows.min().item())
                    print("Max sum:", total_prob_over_rows.max().item())
                
                # Convert to numpy for plotting
                att_map = att_map.numpy()
                
                # Create a heatmap of the attention map
                plt.figure(figsize=(10, 8))
                plt.imshow(att_map, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Attention Map - Layer {layer_idx}, Head {head_idx}')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                
                # Save the plot
                plt.savefig(f'attention_map_layer_encoder{layer_idx}_head{head_idx}.png')
                plt.close()

    def sanity_check02(self, sentence, block_size):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)
        
        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)
        
        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)
        
        # Process the input tensor through the encoder model
        _, _, attention_maps = self.model(input_tensor)
        
        # Display the number of attention maps
        print("Number of layers with attention maps:", len(attention_maps))
        
        # Visualize and save the attention maps
        for layer_idx, layer_attention in enumerate(attention_maps):
            # layer_attention shape is [num_heads, batch_size, seq_len, seq_len]
            num_heads = layer_attention.shape[0]
            
            for head_idx in range(num_heads):
                # Extract attention weights for this head
                # Remove batch dimension and convert to tensor
                att_map = layer_attention[head_idx, 0].detach().cpu()
                
                # Important: Check if the attention probabilities sum to 1 over rows
                total_prob_over_rows = torch.sum(att_map, dim=-1)  # Sum over the key dimension
                
                if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                    print(f"Layer {layer_idx}, Head {head_idx}")
                    print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                    print("Total probability over rows:", total_prob_over_rows.numpy())
                    print("Min sum:", total_prob_over_rows.min().item())
                    print("Max sum:", total_prob_over_rows.max().item())
                
                # Convert to numpy for plotting
                att_map = att_map.numpy()
                
                # Create a heatmap of the attention map
                plt.figure(figsize=(10, 8))
                plt.imshow(att_map, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Attention Map - Layer {layer_idx}, Head {head_idx}')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                
                # Save the plot
                plt.savefig(f'attention_map_layer_decoder{layer_idx}_head{head_idx}.png')
                plt.close()
            

class AttentionVisualizer:
    def __init__(self, model):
        self.model = model
        
    def get_attention_maps(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention maps from model forward pass."""
        with torch.no_grad():
            _, _, attention_maps = self.model(input_ids)
        return attention_maps

    def plot_alibi_slopes(self, num_heads: int, max_len: int = 100) -> None:
        """Visualize ALiBi slopes for each attention head."""
        plt.figure(figsize=(12, 6))
        for head_idx in range(num_heads):
            m = 2 ** -(8 / num_heads) * (head_idx + 1)
            positions = np.arange(max_len)
            bias = -m * positions
            plt.plot(positions, bias, label=f'Head {head_idx + 1}')
        
        plt.title('ALiBi Slopes Across Different Heads')
        plt.xlabel('Position')
        plt.ylabel('Bias Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_attention_heatmap(self, 
                             attention_weights: torch.Tensor,
                             layer: int,
                             head: int,
                             tokens: Optional[List[str]] = None,
                             title: Optional[str] = None) -> None:
        """Plot attention heatmap for a specific layer and head."""
        # Extract attention weights for specified layer and head
        attn = attention_weights[layer][0, head].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        if tokens is None:
            tokens = [f't{i}' for i in range(attn.shape[0])]
            
        sns.heatmap(attn,
                   xticklabels=tokens,
                   yticklabels=tokens,
                   cmap='viridis',
                   vmin=0,
                   vmax=1)
        
        if title is None:
            title = f'Attention Pattern - Layer {layer + 1}, Head {head + 1}'
        plt.title(title)
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_aggregated_attention(self,
                                attention_weights: torch.Tensor,
                                layer: int,
                                tokens: Optional[List[str]] = None,
                                aggregation: str = 'mean') -> None:
        """Plot aggregated attention patterns across all heads for a specific layer."""
        # Get attention weights for specified layer
        layer_attention = attention_weights[layer][0]  # [num_heads, seq_len, seq_len]
        
        if aggregation == 'mean':
            aggregated_attn = layer_attention.mean(dim=0).cpu().numpy()
        elif aggregation == 'max':
            aggregated_attn = layer_attention.max(dim=0)[0].cpu().numpy()
        else:
            raise ValueError("aggregation must be either 'mean' or 'max'")
            
        plt.figure(figsize=(10, 8))
        if tokens is None:
            tokens = [f't{i}' for i in range(aggregated_attn.shape[0])]
            
        sns.heatmap(aggregated_attn,
                   xticklabels=tokens,
                   yticklabels=tokens,
                   cmap='viridis',
                   vmin=0,
                   vmax=1)
        
        plt.title(f'{aggregation.capitalize()} Attention Pattern - Layer {layer + 1}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_attention_statistics(self,
                                attention_weights: torch.Tensor,
                                layer: int) -> None:
        """Plot statistical properties of attention patterns for a specific layer."""
        layer_attention = attention_weights[layer][0]  # [num_heads, seq_len, seq_len]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Mean attention weight per position
        mean_weights = layer_attention.mean(dim=0).mean(dim=0).cpu().numpy()
        axes[0].plot(mean_weights)
        axes[0].set_title('Mean Attention by Position')
        axes[0].set_xlabel('Position')
        axes[0].set_ylabel('Mean Attention Weight')
        axes[0].grid(True)
        
        # Plot 2: Attention entropy per head
        entropy = -(layer_attention * torch.log(layer_attention + 1e-10)).sum(dim=-1).mean(dim=-1)
        axes[1].bar(range(layer_attention.shape[0]), entropy.cpu().numpy())
        axes[1].set_title('Attention Entropy by Head')
        axes[1].set_xlabel('Head')
        axes[1].set_ylabel('Entropy')
        axes[1].grid(True)
        
        # Plot 3: Attention sparsity per head
        sparsity = (layer_attention < 0.01).float().mean(dim=-1).mean(dim=-1)
        axes[2].bar(range(layer_attention.shape[0]), sparsity.cpu().numpy())
        axes[2].set_title('Attention Sparsity by Head')
        axes[2].set_xlabel('Head')
        axes[2].set_ylabel('Sparsity (% near zero)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

def visualize_alibi_patterns(model, input_text: str, tokenizer) -> None:
    """Comprehensive visualization of ALiBi attention patterns."""
    visualizer = AttentionVisualizer(model)
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Get attention maps
    attention_maps = visualizer.get_attention_maps(input_ids)
    
    # 1. Plot ALiBi slopes
    visualizer.plot_alibi_slopes(num_heads=model.blocks[0].sa.heads[0].key.out_features)
    
    # 2. Plot attention patterns for each layer
    for layer_idx in range(len(model.blocks)):
        # Plot individual head patterns
        for head_idx in range(len(model.blocks[layer_idx].sa.heads)):
            visualizer.plot_attention_heatmap(
                attention_maps[layer_idx],
                layer_idx,
                head_idx,
                tokens=tokens
            )
        
        # Plot aggregated attention
        visualizer.plot_aggregated_attention(
            attention_maps[layer_idx],
            layer_idx,
            tokens=tokens
        )
        
        # Plot attention statistics
        visualizer.plot_attention_statistics(
            attention_maps[layer_idx],
            layer_idx
        )


