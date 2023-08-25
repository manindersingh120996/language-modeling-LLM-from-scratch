import re
import torch
import matplotlib.pyplot as plt
%matplotlib inline


dataset = open('dataset.txt','r',encoding='utf8').read().splitlines()

# Define a regular expression pattern to match English letters
english_letters_pattern = re.compile(r'^[a-zA-Z\s]+$')

# Filter names based on the pattern
dataset = [name for name in dataset if english_letters_pattern.match(name)]

print(dataset[:5],len(dataset))

print(min(len(w) for w in dataset),max(len(w) for w in dataset))

# Working with Bigram Model First (Looking at only two words at a time)
# - very week model but good place to start with

unique_characters = set()
for w in dataset:
    unique_characters.update(list(w))
    
unique_characters = sorted(unique_characters)
print(unique_characters,'\n'*2,f"Number of uniqe characters in Train Dataset: {len(unique_characters)}")

# calculating statistics of bigram words occurance
b_count = {}
for w in dataset:
    #creating special character to denote the start and end of generation
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1,ch2 in zip(chs,chs[1:]):
        bigram = (ch1,ch2)
        # check if bigram is there in list and add 1 else intialise it with 0 
        b_count[bigram] = b_count.get(bigram,0) + 1
#         print(ch1,ch2)
#     print("\n")

# sorting more occureance of bigram to least
print(sorted(b_count.items(), key = lambda kv:-kv[1]))

# storing dictionary as two dimensional array (using pytorch to do tensor operations)

# creatging array of unique elemkents + 2 unique charcters'<Start>' and '<End>' and space charcter as well
N = torch.zeros((len(unique_characters)+2,len(unique_characters)+2),
dtype=torch.int32)

# creating lookup table
#mapping string characters to number 
unique_characters.insert(0,'<S>') #= '<S>'
unique_characters.insert(0,'<E>') #= '<E>'
stoi = {s:i for i,s in enumerate(unique_characters)}
# len(stoi)

# for keeping special characters at the end
# stoi['<S>'] = len(stoi)
# stoi['<E>'] = len(stoi)
# len(stoi),stoi

for w in dataset:
    #creating special character to denote the start and end of generation
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1,ch2 in zip(chs,chs[1:]):
        bigram = (ch1,ch2)
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] +=1
        # check if bigram is there in list and add 1 else intialise it with 0 

#inversing string to integer dictionary
itos = {i:s for s,i in stoi.items()}
# itos

plt.figure(figsize=(36,36))
plt.imshow(N,cmap='Blues')
for i in range(len(itos)):
    for j in range(len(itos)):
#         chstr = itos[i] +','+itos[j]
        chstr = itos[i] +itos[j]
        plt.text(j,i,chstr,ha="center",va='bottom',color='gray',size=15)
        plt.text(j,i,N[i,j].item(),ha="center",va='top',color='red',size=13)
plt.axis('off');

# N[0] = 1
#creating smoothing of dataset by adding 1 to all the rows because aster <E> there will be no character
# in order to cover that error of -inf
N += 1

# creating normalised version of vector N
# adding fake count of 1 to model in order to avoid -inf in negative 
# log likelihood,also called as model smoothing
P = (N).float()   
P /= P.sum(1,keepdim=True) # uses the concept of broadcasting for operation of division

g = torch.Generator().manual_seed(122)
# starting Index
ix = 0



for i in range(20):
    print(i+1 , ':',end=' ')
    while True:
        # selecting that row
#         p = N[ix].float()
#         normalising that row for probability distribution for for feeding to multinomial for char generation
#         p = p/p.sum()
        p = P[ix]
        ix = torch.multinomial(p,1,replacement=True,generator=g).item()
        if ix == 0:
            break
        else:
            print(itos[ix],end='')
    print('')
    
    # below code was creating case of randomly generating new sentences but it was used to overcome -inf case of
    # not having characters after <E>, but commented out after adding 1 to all the rows
#     ix = torch.randint(2, len(unique_characters), (1,))

# - perimiter P is the parameter which our bigram model learned.
# - using this P we can calculate the quality of model
# - minimising negative log likelihood, maximising log likelihood,minimizing the average the normalised negative log likelihood

#evaluating quality of model using log likelihood
log_likelihood = 0.0
n = 0
for w in dataset:
    #creating special character to denote the start and end of generation
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1,ch2 in zip(chs,chs[1:]):
        bigram = (ch1,ch2)
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n+=1
#         print(f'{ch1}{ch2}:{prob:.4f} {logprob:.4f}')
print(f'{log_likelihood=}')
negative_log_likelihood = -log_likelihood
print(f'{negative_log_likelihood=}')
print(f'{negative_log_likelihood/n}')

