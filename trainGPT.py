import numpy
# read 'input.txt' file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print length of text in 'input.txt' file
print("Length of text in document: ", len(text))

# print first 1000 characters in the file
print(text[:1000])

# created sorted list of all unique characters in this set (first 1000 characters)
chars = sorted(list(set(text)))
# All unique characters used in text set
vocab_size = len(chars)
print("Vocabulary size: ", vocab_size)
print("All vocab in set:")
print(''.join(chars))
print()

# create a tokenization protocol/mapping for characters to integer
strToInt = {char:int for int, char in enumerate(chars)}
intToStr = {int:char for int, char in enumerate(chars)}
encode = lambda str: [strToInt[c] for c in str] # encoding mechanism: takes a string, outputs a list of integers
decode = lambda int: ''.join([intToStr[i] for i in int]) # decoding mechanism: takes a list of integers, outputs a string

print(encode("hello"))
print(decode(encode("hello")))

# encode the entire text dataset using pyTorch and store it into a torch.Tensor
import torch
data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape, data.type)
print(data[:1000]) # the 1000 character set we looked at earlier will be read by the GPT like this

# split the data into training data and validation data
split = int(0.9*len(data)) # split data 90/10
# first 90% is used for training, other 10% is used for validation
training_data = data[:split]
validation_data = data[split:]

#block size is the number of previous tokens that will be taken into consideration when making a prediction
block_size = 8
var = training_data[:block_size + 1]

# creates context-target pairs that train the model efficiently
x = training_data[:block_size]
y = training_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is {target}")

torch.manual_seed(1337)
batch_size = 4 #number of independent sequences processed in parallel
block_size = 8 # maximum context length for predictions



def getBatch(split):
    #decide whether the data is used for training or validation
    data = training_data if split == 'train' else validation_data
    #generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = getBatch('train')
print('inputs: ')
print(xb.shape)
print(xb)
print('targets: ')
print(yb.shape)
print(yb)

print('-----')

for batch in range(batch_size):
    for time in range(block_size):
            #what the model reads
            context = xb[batch, :time+1]
            #what the model predicts
            target = yb[batch, time]
            print(f"when input is {context.tolist()} the target is {target}")


# pytorch module that implements the bigram language model

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token off the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):

        #idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)
        # B = batch size, T = block size, C = vocab size
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            #pyTorch's cross-entropy expects: (logits: [N, C], targets: [N])
            # so B and T are flattened into (B*T, C)
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            #Compares predicted probabilities vs actual targets
            #returns a single value that represents the accuracy of the model
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1)
            # randomly picks a token based on probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
print(decode(m.generate(idx = torch.zeros((1, 1), dtype = torch.long), max_new_tokens = 100) [0].tolist()))
