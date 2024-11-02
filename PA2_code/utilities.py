
import matplotlib.pyplot as plt
import torch
import numpy as np

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
            


