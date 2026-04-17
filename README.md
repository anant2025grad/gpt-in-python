**Bigram Language Model (PyTorch)**

A character-level language model built from scratch using PyTorch. This project implements a basic bigram-based neural network that learns to predict the next character given the current one, trained on raw text data.

**Overview**

This project explores the fundamentals of language modeling by implementing a simple bigram model using neural networks. Instead of using complex architectures, the model learns directly from character transitions in text.

At its core, the model estimates:

    P(next_token | current_token)

Despite its simplicity, it demonstrates key machine learning concepts such as:

- Tokenization
- Embeddings
- Cross-entropy loss
- Gradient-based optimization
- Autoregressive text generation

**How It Works**
1. Data Processing

  - Reads raw text from input.txt
  - Builds a vocabulary of unique characters
  - Encodes characters into integers
  - Splits data into training (90%) and validation (10%)

2. Model Architecture

The model uses a single PyTorch embedding layer:

    nn.Embedding(vocab_size, vocab_size)

  This acts as a lookup table:
  - Each token maps directly to logits for the next token
  - No hidden layers — purely probabilistic mapping

3. Training
  - Optimizer: Adam
  - Loss function: Cross-Entropy
  - Batch sampling with random subsequences
  - Periodic evaluation on validation data

        loss = F.cross_entropy(logits, targets)

4. Text Generation
  The model generates text autoregressively:
    - Start with an initial token
    - Predict next-token probabilities
    - Sample from distribution
    - Append token and repeat
  
**Example Output**

After training, the model generates character-level text such as:

*Thou shalt not speak of dreams that wander beyond the night...*

(Note: Output quality depends on dataset and training time)

Training Insights
- Train Loss vs Validation Loss is tracked to monitor learning and generalization
- Loss represents the model’s confidence in predicting the correct next character
- Lower loss = better predictions
Key Concepts Demonstrated
- Batch processing (B, T, C tensor structure)
- Sequence modeling
- Gradient descent & backpropagation
- Probabilistic sampling (torch.multinomial)
- Model evaluation without gradient tracking (@torch.no_grad())
Future Improvements

This project is actively being expanded. Planned upgrades include:
- Self-attention mechanism (Transformer blocks)
- Multi-head attention
- Positional embeddings
- Deeper network architecture
- Improved sampling (temperature, top-k)
- Larger datasets for better generation quality

How to Run
- Install dependencies:

      pip install torch

- Add your dataset:
- Place a text file named input.txt in the project directory
- Run the script:

      python bigramModel.py

**Why This Project Matters**

This project builds intuition for how modern language models (like GPT) work — starting from the simplest possible foundation.
It demonstrates:
- Strong understanding of ML fundamentals
- Ability to implement models from scratch
- Clear progression toward more advanced architectures


Implemented by Anant V. Palve
Based on Andrej Karpathy's 'Let's build GPT: from scratch, in code, spelled out.' YouTube video.

