# Transformer Blocks

**Description**  
This project implements a Encoder and Decoder Model

## Project Structure

```plaintext
├── main.py          # Contains usage of dataset loading, training loops. The feedforward classifier can also be implemented here.
├── utilities.py     # Contains helper functions for sanity checking your attention implementation
├── tranformer.py    # Implement the encoder and decoder classes
├── tokenizer.py     # Contains a simple tokenizer class that tokenizes the input text into words
├── sparse.py        # Implementation of Sparse Attention Patterns
├── disentagled.py   # Implementation of Disentangled Attention Patterns
├── alibi.py         # Implementation Attention with Linear Biases
└── README.md        # Project overview and instructions
├── dataset.py       # PyTorch Dataset classes for the classification and language modeling tasks
```
All the png files generated are attatched. The relevant png files are disccused in the report.

## Dependencies

- **Python 3.6+**
- **PyTorch**
- **NumPy**
- **Anaconda**

## Set up the Environment

1. **Create a virtual environment and install dependency** using Anaconda (recommended) or `venv`:

   ```bash
   conda create -n transformer-blocks python=3.8
   conda activate transformer-blocks
   conda install pytorch numpy -c pytorch

## Model Run

To run the models, use the following commands:
    python main.py 

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.