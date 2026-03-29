import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset

n_embed=32
head_size=32

dataset = load_dataset("roneneldan/TinyStories", split="train")
#print(dataset[0])
text="\n".join(dataset[i]["text"]for i in range(5000))
print(len(text))

chars=sorted(list(set(text)))
vocab_size = len(chars)

print(len(chars))
#print(chars) prints all the unique characters in the text
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
#print(data.shape, data.dtype)
#print(data[:100]) prints the first 100 encoded characters as integers in tensor form

n=int(0.9*len(data))
train_data=data[:n] # split the data into training and validation sets
val_data=data[n:]

block_size=32 # the number of characters we want to feed into the model at once
x=train_data[:block_size] # the input to the model will be the first 32 encoded characters
y=train_data[1:block_size+1] # the target output will be the next 32 encoded characters, which is the input shifted by one character
#print("x:", x)  
#print("y:", y)

batch_size=4 # the number of sequences we want to feed into the model at once
def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size, (batch_size,)) # randomly select batch_size starting indices for the sequences
    x=torch.stack([data[i:i+block_size] for i in ix])     # create a batch of input sequences by slicing the data at the selected indices
    y=torch.stack([data[i+1:i+block_size+1] for i in ix]) # create a batch of target sequences ,shifted by one character
    return x, y 

xb, yb=get_batch("train")
#print("xb:", xb) prints a batch of input sequences as integers in tensor form
#print("yb:", yb)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  
        q = self.query(x) 
        wei=q @ k.transpose(-2, -1) * head_size**-0.5
        wei=wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) #mask to ensure model only attends to previous tokens
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embed)
        self.sa_head=Head(head_size) 
        self.lm_head=nn.Linear(n_embed, vocab_size) # create a linear layer for the language model head
    
    def forward(self, idx, targets=None):
        x=self.token_embedding_table(idx) 
        x=self.sa_head(x)
        logits=self.lm_head(x)

        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape 
            logits=logits.view(B*T,C) # reshape the logits to have shape (B*T, C) for computing the loss
            targets=targets.view(B*T) 
            loss=F.cross_entropy(logits, targets) # compute the cross-entropy loss between the logits and targets
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) 
            logits = logits[:, -1, :] # focus on the last time step's logits
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) # append the new token index to the input indices
        return idx
    
model = BigramLanguageModel(vocab_size)
xb, yb = get_batch('train')
logits, loss = model(xb, yb)
#print(loss.item())

idx = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(idx, max_new_tokens=200)
#print(decode(generated[0].tolist()))

# Creating pytorch optimizer and training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("loss=", loss.item())

print(decode(model.generate(idx, max_new_tokens=200)[0].tolist()))


    
