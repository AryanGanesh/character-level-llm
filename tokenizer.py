from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories", split="train")
print(dataset[0])
text="\n".join(dataset[i]["text"]for i in range(5000))
print(len(text))
chars=sorted(list(set(text)))
print(len(chars))
print(chars)
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

print(decode(encode("hello world")))

