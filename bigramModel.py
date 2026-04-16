import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200

torch.manual_seed(1337)

# read 'input.txt' file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print length of text in 'input.txt' file
print('')
print("Length of text in document: ", len(text))
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


# encode the entire text dataset using pyTorch and store it into a tensor
data = torch.tensor(encode(text), dtype = torch.long)
split = int(0.9*len(data)) # split the data into training data and validation data 90/10, first 90% is used for training, other 10% is used for validation
training_data = data[:split]
validation_data = data[split:]

def getBatch(split):
    #decide whether the data is used for training or validation
    data = training_data if split == 'train' else validation_data
    #generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@torch.no_grad() #context manager that is useful if '.backward()' is not going to be used; good for memory efficiency
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = getBatch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# pytorch module that implements the bigram language model
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

# create a pyTorch optimizer by implementing AdamW
optimizer = torch.optim.Adam(m.parameters(), lr = 1e-3)

for step in range(max_iters):

    if step % eval_interval == 0:
        loss = estimate_loss()
        print(f"step {step}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f} ")

    # sample of a bunch of data
    input, expected_output = getBatch('train')

    # evaluate the loss
    logits, loss = m(input, expected_output)
    # zero out all the gradients
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype = torch.long)
print(decode(m.generate(context, max_new_tokens = 500) [0].tolist()))