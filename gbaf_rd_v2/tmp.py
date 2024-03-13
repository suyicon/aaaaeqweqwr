import math
from time import time

import numpy.random
import torch
import numpy as np
from torch.distributions import Categorical
from parameters import args_parser

args = args_parser()
n = int(args.K*0.5+1)
prob = torch.tensor([math.comb(args.K, k)*np.power(0.5,args.K) for k in range(n,args.K+1)])
print(prob)
matrix = torch.randint(0,2,(4096,48))
mask = torch.zeros_like(matrix).bool()
dist = Categorical(prob)
flips = dist.sample((4096, 1))+25
for i in range(matrix.shape[0]):
    indices = torch.randperm(matrix.shape[1])[:flips[i]]
    mask[i, indices] = True
print(matrix)
matrix = torch.where(mask,matrix^1,matrix)
print(matrix)
print(matrix.shape)
# bulls = (seq_binary == bVec_binary).sum(dim=1)
# a = sum(bulls[:] > 24)
# while sum(bulls[:] < threshold) > 0:
#     seq_binary[bulls[:] < threshold, :] = torch.randint(0, 2, (sum(bulls[:] < threshold), args.K))
#     bulls = (seq_binary == bVec_binary).sum(dim=1)
# time2 = time.time()


# print(a)
# print(b)
