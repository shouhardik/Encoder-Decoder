
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
        _, attention_maps = self.model(input_tensor)
        
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
            
    def sanity_check03(self, sentence, block_size):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)
        
        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * max(0, block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)
        
        # Display input tensor shape and tokens for debugging
        #print("Input tensor shape:", input_tensor.shape)
        #print("Tokens:", [self.tokenizer.decode([id]) for id in padded_sentence[:len(wordids)]])
        
        # Process the input tensor through the model
        _, loss, attention_maps = self.model(input_tensor, input_tensor)  # Note: for decoder, we need both input and target
        
        # Display the number of attention maps
        print("Number of layers with attention maps:", len(attention_maps))
        
        # Create a figure with subplots for all attention heads
        n_layers = len(attention_maps)
        n_heads = attention_maps[0].shape[0]
        
        for layer_idx, layer_attention in enumerate(attention_maps):
            fig, axes = plt.subplots(1, n_heads, figsize=(20, 5))
            if n_heads == 1:  # Handle the case when there's only one head
                axes = [axes]
            fig.suptitle(f'Layer {layer_idx} Attention Maps')
            
            for head_idx in range(n_heads):
                # Extract attention weights for this head
                # Remove batch dimension and convert to numpy
                att_map = layer_attention[head_idx, 0, :len(wordids), :len(wordids)].detach().cpu().numpy()
                
                # Create heatmap with original 'hot' colormap
                im = axes[head_idx].imshow(att_map, cmap='hot', interpolation='nearest')
                axes[head_idx].set_title(f'Head {head_idx}')
                
                # Add token labels
                token_labels = [self.tokenizer.decode([id]) for id in padded_sentence[:len(wordids)]]
                axes[head_idx].set_xticks(range(len(token_labels)))
                axes[head_idx].set_yticks(range(len(token_labels)))
                axes[head_idx].set_xticklabels(token_labels, rotation=45, ha='right')
                axes[head_idx].set_yticklabels(token_labels)
                
                # Add colorbar
                plt.colorbar(im, ax=axes[head_idx])
            
            plt.tight_layout()
            plt.savefig(f'attention_map_layer_decoder{layer_idx}.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # Validate attention weights
            for head_idx in range(n_heads):
                att_map = layer_attention[head_idx, 0].detach().cpu()
                row_sums = torch.sum(att_map, dim=-1)
                
                # Check if attention weights sum to 1
                if not torch.allclose(row_sums[:len(wordids)], torch.ones_like(row_sums[:len(wordids)]), atol=1e-5):
                    print(f"Warning: Layer {layer_idx}, Head {head_idx} attention weights don't sum to 1")
                    print(f"Row sums:")
                    for i, sum_val in enumerate(row_sums[:len(wordids)]):
                        print(f"Token {token_labels[i]}: {sum_val.item():.6f}")