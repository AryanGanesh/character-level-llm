import torch
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories", split="train")
print(dataset[0])
text="\n".join(dataset[i]["text"]for i in range(5000))
print(len(text))
chars=sorted(list(set(text)))
print(len(chars))
#print(chars) ptints all the unique characters in the text
d = {}
for i, ch in enumerate(chars):
    d[ch] = i

def encode(string):
    result=[]
    for ch in string:
        result.append(d[ch])
    return result

def decode(list_of_ints):
    result=""
    for i in list_of_ints:
        result+=chars[i]
    return result

# print(decode(encode("hello world"))) test to check if the encode and decode functions are working correctly

data=torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
#print(data[:100]) prints the first 100 encoded characters as integers in tensor form

n=int(0.9*len(data))
train_data=data[:n] # split the data into training and validation sets
val_data=data[n:]

block_size=32 # the number of characters we want to feed into the model at once
x=train_data[:block_size] # the input to the model will be the first 32 encoded characters
y=train_data[1:block_size+1] # the target output will be the next 32 encoded characters, which is the input shifted by one character
print("x:", x)  #prints the first 32 encoded characters as integers in tensor form
print("y:", y)

batch_size=4 # the number of sequences we want to feed into the model at once
def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size, (batch_size,)) # randomly select batch_size starting indices for the sequences
    x=torch.stack([data[i:i+block_size] for i in ix])     # create a batch of input sequences by slicing the data at the selected indices
    y=torch.stack([data[i+1:i+block_size+1] for i in ix]) # create a batch of target sequences ,shifted by one character
    return x, y 

xb, yb=get_batch("train")
print("xb:", xb) 
print("yb:", yb)