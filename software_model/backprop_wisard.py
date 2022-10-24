#!/usr/bin/false

import torch
import torch.nn as nn

import numpy as np
from os import urandom
from scipy.stats import norm

# Computes hash functions within the H3 family of integer-integer hashing functions,
#  as described by Carter and Wegman in the paper "Universal Classes of Hash Functions"
# Inputs:
#  x:         A 2D tensor (dxb) consisting of d b-bit values to be hashed, expressed as bitvectors
#  hash_vals: A 2D tensor (hxb) consisting of h sets of b int64s, representing random constants to compute h unique hashes
# Returns: A 2D tensor (dxh) of int64s, representing the results of the h hash functions on the d input values
def h3_hash(x, hash_vals):
    # Choose between hash values and 0 based on corresponding input bits
    selected_entries = torch.einsum("hb,db->bdh",hash_vals,x) # Surprisingly, this is faster than a conditional lookup using e.g. torch.where

    # Perform XOR reduction along input (b) axis
    reduction_result = torch.zeros((x.shape[0], hash_vals.shape[0]), dtype=torch.int64, device=x.device)
    for i in range(hash_vals.shape[1]):
        reduction_result.bitwise_xor_(selected_entries[i]) # In-place XOR
    return reduction_result

# Generates random constants for H3 hash functions
# Inputs:
#  unit_inputs:  Number of inputs to each Bloom filter (b)
#  unit_entries: Number of entries in each Bloom filter's LUT
#  unit_hashes:  Number of unique H3 functions to generate parameters for (h)
# Returns: A 2D tensor (hxb) of random int64s
def generate_h3_values(unit_inputs, unit_entries, unit_hashes):
    assert(np.log2(unit_entries).is_integer())
    shape = (unit_hashes, unit_inputs)
    values = torch.from_numpy(np.random.randint(0, unit_entries, shape))
    return values

# Performs sign-based binarization, using the straight-through estimator with clamping for gradients
# Derived from the method described in "Binarized Neural Networks" by Hubara et al. (NeurIPS 2016):
#  https://proceedings.neurips.cc/paper/2016/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf
class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        outp = (inp >= 0).float() # 0/1 binarization
        return outp
   
    @staticmethod
    def backward(ctx, grad_outp):
        # Straight-through; cancel gradient where |inp| > 1
        inp, = ctx.saved_tensors
        grad_inp = torch.where(inp.abs() <= 1, grad_outp, torch.tensor(0.0).to(inp.device))
        return grad_inp

# Implementation of WiSARD model with differentiable table entries
class BackpropWiSARD(nn.Module):
    def __init__(self, num_inputs, num_classes, unit_inputs, unit_entries, unit_hashes, bits_per_input, train_data):
        super().__init__()
        self.num_inputs, self.num_classes, self.unit_inputs, self.unit_entries, self.unit_hashes = num_inputs, num_classes, unit_inputs, unit_entries, unit_hashes
        self.bits_per_input = bits_per_input
        
        # Compute encoding thresholds
        train_mean = train_data.mean(axis=0)
        train_std = train_data.std(axis=0)
        std_skews = torch.Tensor([norm.ppf((i+1)/(bits_per_input+1)) for i in range(bits_per_input)])
        thresholds = ((std_skews.view(-1, 1).repeat(1, num_inputs) * train_std) + train_mean).T

        thresholds = torch.maximum(thresholds.T, train_data.kthvalue(2, axis=0).values).T # Thresholds should be at least second-smallest value
        thresholds = torch.minimum(thresholds.T, train_data.amax(axis=0)).T # Thresholds should be at most largest value
        self.thresholds = nn.Parameter(thresholds, requires_grad=False)
       
        # Get input information and number of filters
        input_bits = int(np.ceil((num_inputs*bits_per_input)/unit_inputs)) * unit_inputs # Total number of inputs to the model, padded to integer # of units
        self.null_bits = input_bits - (num_inputs*bits_per_input) # Number of extra bits needed to make # model inputs an integer multiple of # unit inputs
        self.filters_per_discrim = input_bits // unit_inputs
       
        # Initialize data tensor (3D tensor - discriminator x filter x entry)
        self.data = nn.Parameter(torch.Tensor(num_classes, self.filters_per_discrim, unit_entries))
        nn.init.uniform_(self.data, -1, 1)

        self.hash_values = nn.Parameter(generate_h3_values(unit_inputs, unit_entries, unit_hashes), requires_grad=False)

        input_order = np.arange(input_bits).astype(int)
        np.random.shuffle(input_order)
        self.input_order = nn.Parameter(torch.from_numpy(input_order).long(), requires_grad=False)

        self.mask = nn.Parameter(torch.ones((num_classes, self.filters_per_discrim)), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(num_classes), requires_grad=False)

    def get_filter_responses(self, x):
        batch_size = x.shape[0]
        
        # Binarize with thermometer encoding
        duplicated = x.view(batch_size, -1, 1).expand(-1, -1, self.bits_per_input)
        binarized = (duplicated >= self.thresholds).byte().reshape(batch_size, -1)

        # Pad inputs to integer multiple of unit size and reorder
        padded = nn.functional.pad(binarized, (0, self.null_bits))
        mapped = padded[:,self.input_order]

        # Hash filter inputs (using H3)
        hash_inputs = mapped.view(batch_size*self.filters_per_discrim, self.unit_inputs)
        hash_outputs = h3_hash(hash_inputs, self.hash_values)

        # Perform table lookup - this requires some weird data munging
        #  1. Reshape hash outputs into 3D discriminator x filter x (batch * hashfunction) tensor, duplicating (expanding) along discriminator axis
        filter_inputs = hash_outputs\
                .view(batch_size, self.filters_per_discrim, self.unit_hashes)\
                .permute(1, 0, 2)\
                .reshape(1, self.filters_per_discrim, -1)\
                .expand(self.num_classes, -1, -1)
        #  2. Implement high-dimensional table lookups as a single gather operation
        flat_lookup = self.data.gather(2, filter_inputs)
        #  3. Restructure 3D discriminator x filter x (batch * hashfunction) tensor into 4D batch x discriminator x filter x hashfunction tensor
        lookup = flat_lookup\
                .view(self.num_classes, self.filters_per_discrim, batch_size, self.unit_hashes)\
                .permute(2, 0, 1, 3)

        bin_lookup = BinarizeFunction.apply(lookup) # Binarize lookup to 0/1
        responses = bin_lookup.amin(axis=-1) # Reduce lookup along hashfunction dimension
        return responses
        
    def forward(self, x):
        responses = self.get_filter_responses(x)

        masked_responses = responses * self.mask
        result = masked_responses.sum(axis=2) + self.bias # Calculate activations for each discriminaton
        return result

    def clamp(self): # Clamp data to [-1, 1]
        self.data.data.clamp_(-1, 1)

