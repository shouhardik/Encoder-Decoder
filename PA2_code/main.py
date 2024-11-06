import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import EncoderModel, DecoderModel
from alibi import EncoderModelALiBi, DecoderModelALiBi
from utilities import Utilities, visualize_alibi_patterns
from sparse import SparseEncoderModel, SparseDecoderModel
from disentagled import DistangledEncoderModel, DisentagledDecoderModel
from rpe import RelativePositionEncoder, RelativePositionDecoder
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
model_dim = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

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



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0  # Initialize total_loss here
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocab_size):
        super().__init__()
        # Encoder components
        self.encoder = EncoderModel(vocab_size)
        
        # Classifier layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # hidden layer
        self.fc2 = nn.Linear(hidden_dim, output_dim) #output layer
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, targets=None):
        encoder_output, attention_maps = self.encoder(x)
        
        # Pass through classifier layers
        x = F.relu(self.fc1(encoder_output))  # Hidden layer with ReLU activation
        x = self.dropout(x) # Dropout applied after the hidden layer
        logits = self.fc2(x) # Output layer producing final logits
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss, attention_maps

def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    model = EncoderModel(tokenizer.vocab_size).to(device)
    #model = EncoderModelALiBi(tokenizer.vocab_size).to(device)
    #model = SparseEncoderModel(tokenizer.vocab_size).to(device)
    #model = DistangledEncoderModel(tokenizer.vocab_size).to(device)
    #model = RelativePositionEncoder(tokenizer.vocab_size).to(device)
    # we can create an Optimizer

    # Initialize classifier (which includes the encoder)
    classifier = Classifier(model_dim, n_hidden, n_output, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)

    utility = Utilities(tokenizer, model)
    sentence = "Sample sentence for  my sanity check for NLP Q1 PA2"
    utility.sanity_check01(sentence, block_size)
  
    decoderModel = DecoderModel(tokenizer.vocab_size).to(device)
    #decoderModel = DecoderModelALiBi(tokenizer.vocab_size).to(device)
    #decoderModel = SparseDecoderModel(tokenizer.vocab_size).to(device)
    #decoderModel = DisentagledDecoderModel(tokenizer.vocab_size).to(device)
    #decoderModel = RelativePositionDecoder(tokenizer.vocab_size).to(device)
    decoder_optimizer = torch.optim.AdamW(decoderModel.parameters(), lr=learning_rate)

    utility = Utilities(tokenizer, decoderModel)
    sent = "Sample sentence for  my sanity check for NLP Q2 PA2"
    utility.sanity_check02(sent, block_size)

    # alibiModel = EncoderModelALiBi(tokenizer.vocab_size).to(device)
    # tokenizer = tokenizer
    # text = "Example input text to analyze"
    # # util = AttentionVisualizer(alibiModel)
    # # util.
    # visualize_alibi_patterns(alibiModel, text, tokenizer)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

        # Load test files correctly
    with open("speechesdataset/test_LM_hbush.txt", 'r', encoding='utf-8') as f:
        hbush_test = f.read()
    with open("speechesdataset/test_LM_obama.txt", 'r', encoding='utf-8') as f:
        obama_test = f.read()
    with open("speechesdataset/test_LM_wbush.txt", 'r', encoding='utf-8') as f:
        wbush_test = f.read()

    # Create datasets with test data
    test_LM_hbush_dataset = LanguageModelingDataset(tokenizer, hbush_test, block_size)
    test_LM_obama_dataset = LanguageModelingDataset(tokenizer, obama_test, block_size)
    test_LM_wbush_dataset = LanguageModelingDataset(tokenizer, wbush_test, block_size)

    test_LM_hbush_loader = DataLoader(test_LM_hbush_dataset, batch_size=batch_size, shuffle=True)

    test_LM_obama_loader = DataLoader(test_LM_obama_dataset, batch_size=batch_size, shuffle=True)

    test_LM_wbush_loader = DataLoader(test_LM_wbush_dataset, batch_size=batch_size, shuffle=True)

    


     # for the classification  task, you will train for a fixed number of epochs like this:

    for epoch in range(epochs_CLS):
        total_loss = 0
        num_batches = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss, attention_map = classifier(xb, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        train_acc = compute_classifier_accuracy(classifier, train_CLS_loader)
        test_acc = compute_classifier_accuracy(classifier, test_CLS_loader)
    
        print(f'Epoch {epoch}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%')

    # After training, compute final accuracies
    final_train_acc = compute_classifier_accuracy(classifier, train_CLS_loader)
    final_test_acc = compute_classifier_accuracy(classifier, test_CLS_loader)
    print(f'Final Train Accuracy: {final_train_acc:.2f}%')
    print(f'Final Test Accuracy: {final_test_acc:.2f}%')

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    total_loss = 0
    num_batches = 0
    train_perplexities = []

    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        
        # LM training code
        _, loss, _ = decoderModel(xb, yb)
        
        # Backward pass
        decoder_optimizer.zero_grad()
        loss.backward()
        decoder_optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (i + 1) % eval_interval == 0:
            avg_loss = total_loss / num_batches
            perplexity_train = compute_perplexity(decoderModel, train_LM_loader, eval_iters)
            train_perplexities.append(perplexity_train)
            print(f'Iteration {i+1}: Perplexity (Training Set): {perplexity_train:.2f}')
            total_loss = 0
            num_batches = 0

    # Print training perplexities
    # print("\nTraining Perplexities:")
    # for iter_num, perplexity in enumerate(train_perplexities, start=1):
    #     print(f"Perplexity at the {iter_num * 100}th iteration on the training set: {perplexity:.2f}")

    # Compute final perplexities on test sets
    perplexity_obama = compute_perplexity(decoderModel, test_LM_obama_loader, eval_iters)
    perplexity_hbush = compute_perplexity(decoderModel, test_LM_hbush_loader, eval_iters)
    perplexity_wbush = compute_perplexity(decoderModel, test_LM_wbush_loader, eval_iters)

    print("\nFinal Test Perplexities:")
    print(f'Perplexity at the 500th iteration on the test_LM_obama.txt: {perplexity_obama:.2f}')
    print(f'Perplexity at the 500th iteration on the test_LM_hbush.txt: {perplexity_hbush:.2f}')
    print(f'Perplexity at the 500th iteration on the test_LM_wbush.txt: {perplexity_wbush:.2f}')

    print("\nLanguage model training completed.")


    



if __name__ == "__main__":
    main()
