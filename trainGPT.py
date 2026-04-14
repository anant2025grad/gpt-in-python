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

block_size = 8
var = training_data[:block_size + 1]