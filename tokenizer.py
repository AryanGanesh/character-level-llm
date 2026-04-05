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

block_size=32                # the number of characters we want to feed into the model at once
x=train_data[:block_size] 
y=train_data[1:block_size+1] # the target output will be the next 32 encoded characters, which is the input shifted by one character
#print("x:", x)  
#print("y:", y)

batch_size=4 # the number of sequences we want to feed into the model at once
def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size, (batch_size,)) # randomly select batch_size starting indices for the sequences
    x=torch.stack([data[i:i+block_size] for i in ix])    
    y=torch.stack([data[i+1:i+block_size+1] for i in ix]) # create a batch of target sequences ,shifted by one character
    return x, y 

xb, yb=get_batch("train")
#print("xb:", xb) prints a batch of input sequences as integers in tensor form
#print("yb:", yb)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  
        q = self.query(x) 
        wei=q @ k.transpose(-2, -1) * self.head_size**-0.5       # compute the attention weights by taking the dot product of the query and key, and scaling by the square root of the head size
        wei=wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) #mask to ensure model only attends to previous tokens
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out    
    
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        self.sa_head = MultiHeadAttention(num_heads, n_embed // num_heads)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = self.sa_head(x) + x  # add the input to the output of the attention head (residual connection)
        x = self.ffwd(x) + x      # add the input to the output of the feedforward network (residual connection)
        return x
    
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embed)
        self.blocks = nn.Sequential(Block(n_embed, num_heads=4), Block(n_embed, num_heads=4), Block(n_embed, num_heads=4))
        #self.ffwd=FeedForward(n_embed)
        self.lm_head=nn.Linear(n_embed, vocab_size) # create a linear layer for the language model head
    
    def forward(self, idx, targets=None):
        x=self.token_embedding_table(idx) 
        x = self.blocks(x)
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
            logits = logits[:, -1, :] 
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