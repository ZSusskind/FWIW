#!/usr/bin/env python3

import torch
import numpy as np
from numba import jit

import os
import sys
import pickle
import lzma

from backprop_wisard import BinarizeFunction
import main_backprop

# Supports sparsity; does not support training

@jit(nopython=True)
def h3_hash(xv, m):
    #selected_entries = np.where(xv, m, 0)
    selected_entries = xv * m # np.where is unsupported in Numba
    #reduction_result = np.bitwise_xor.reduce(selected_entries, axis=1)
    reduction_result = np.zeros(m.shape[0], dtype=np.int64) # ".reduce" is unsupported in Numba
    for i in range(m.shape[1]):
        reduction_result ^= selected_entries[:,i]
    return reduction_result

class BloomFilter:
    def __init__(self, data, hash_values):
        assert(isinstance(data, np.ndarray))
        assert(isinstance(hash_values, np.ndarray))
        self.data, self.hash_values = data, hash_values

    # Implementation of the check_membership function
    # Coding in this style (as a static method) is necessary to use Numba for JIT compilation
    @staticmethod
    @jit(nopython=True)
    def __check_membership(xv, hash_values, data, soft_error_rate):
        hash_results = h3_hash(xv, hash_values)
        if soft_error_rate > 0.0:
            # Take XOR of (binary) results with random binary vector
            # Probability of a given entry in the random vector being 1 is given by soft_error_rate
            soft_error_vector = np.random.binomial(1, soft_error_rate, hash_results.size)
            hash_results ^= soft_error_vector
        return data[hash_results].all()

    def check_membership(self, xv, soft_error_rate=0.0):
        return BloomFilter.__check_membership(xv, self.hash_values, self.data, soft_error_rate)

class Discriminator:
    def __init__(self, data, mask, hash_values):
        self.nonsparse_filters = {}
        self.num_filters = len(data)
        for i in range(self.num_filters):
            if mask[i]:
                self.nonsparse_filters[i] = BloomFilter(data[i], hash_values)

    def predict(self, xv, soft_error_rate=0.0):
        filter_inputs = xv.reshape(self.num_filters, -1) # Divide the inputs between the filters
        response = 0
        for idx, inp in enumerate(filter_inputs):
            if idx in self.nonsparse_filters:
                response += int(self.nonsparse_filters[idx].check_membership(inp,soft_error_rate))
        return response

class WiSARD:
    def __init__(self, data, mask, pad, hash_values, input_order):
        self.num_classes, _, self.unit_entries = data.shape
        self.unit_hashes, self.unit_inputs = hash_values.shape
        self.num_filters = mask.shape[1]
        self.discriminators = [Discriminator(data[i], mask[i], hash_values) for i in range(self.num_classes)]
        self.input_order = input_order
        self.pad = pad

    def predict(self, xv, soft_error_rate=0.0):
        padded = np.pad(xv, (0, self.pad))
        mapped = padded[self.input_order]
        responses = np.array([d.predict(mapped, soft_error_rate) for d in self.discriminators], dtype=int)
        return responses

class EnsembleWiSARD:
    def __init__(self, data, mask, pad, hash_values, input_order, bias, num_inputs, bits_per_input):
        self.num_models = len(data)
        self.wisard_models = [WiSARD(data[i], mask[i], pad[i], hash_values[i], input_order[i]) for i in range(self.num_models)]
        self.bias = bias
        self.num_inputs = num_inputs
        self.bits_per_input = bits_per_input

    def predict(self, xv, soft_error_rate=0.0):
        responses = np.array([w.predict(xv, soft_error_rate) for w in self.wisard_models], dtype=int)
        total_responses = responses.sum(axis=0)
        if self.bias is not None:
            total_responses += self.bias
        return np.argmax(total_responses)

#unused_inputs, input_remap = get_unused_inputs(model)
def get_unused_inputs(model):
    num_inputs = model.thresholds.shape[0]
    used = np.zeros(num_inputs)
    bits_per_input = model.bits_per_input
    max_input = model.input_order.max().item()
    flat_mask = model.mask.any(axis=0)
    for i in range(model.filters_per_discrim):
        if flat_mask[i]:
            for j in range(i*model.unit_inputs, (i+1)*model.unit_inputs):
                input_idx = model.input_order[j].item() // bits_per_input
                if input_idx < num_inputs: # Need to worry about zero padding at end...
                    used[input_idx] = 1

    unused_inputs = np.where(used == 0)[0]
    input_remap = np.zeros(max_input+1, dtype=int)
    input_idx = 0
    remap_idx = 0
    for i in used:
        for j in range(bits_per_input):
            if i:
                input_remap[input_idx] = remap_idx
                remap_idx += 1
            else:
                input_remap[input_idx] = -1
            input_idx += 1
    for i in range(num_inputs*bits_per_input, max_input+1): # Handle zero pad inputs
        input_remap[input_idx] = remap_idx
        input_idx += 1
        remap_idx += 1
    return unused_inputs, input_remap


def finalize_model(model, unused_inputs, input_remap, adjust_bias = True):
    num_inputs = model.thresholds.shape[0] - len(unused_inputs)
    model_data = BinarizeFunction.apply(model.data).detach().numpy().astype(bool)
    model_mask = model.mask.numpy()
    model_pad = model.null_bits
    model_hash_values = model.hash_values.numpy()
    model_input_order = input_remap[model.input_order.cpu().numpy()]

    # Correct for the fact that values are -1/1 during training, but 0/1 during inference
    # t = p + n; a = p - n + b
    # a = 2p + (b - t)
    model_bias = ((model.bias.numpy() - model.filters_per_discrim)/2).round().astype(int)
    model_bias -= model_bias.min()

    bits_per_input = model.thresholds.shape[1]
    model = EnsembleWiSARD(\
            [model_data], [model_mask], [model_pad],\
            [model_hash_values], [model_input_order], model_bias,\
            num_inputs, bits_per_input)
    return model

def thermometer_encode_dataset(dset, thresholds, unused_inputs):
    num_inputs = thresholds.shape[0] - len(unused_inputs)
    kept_inputs = torch.Tensor([i for i in range(thresholds.shape[0]) if i not in unused_inputs]).long()
    kept_thresholds = thresholds[kept_inputs]
    bits_per_input = thresholds.shape[1]
    bytes_per_sample = int(np.ceil((num_inputs * bits_per_input) / 8))
    encoded_dset = np.array(bytes_per_sample).astype(">u4").tobytes() # Header - big-endian 32-bit value
    encoded_dset += np.array([0]).astype("u1").tobytes() # Reserved for future use
    encoded_dset += np.array(bits_per_input).astype("u1").tobytes()
    for data, label in dset:
        duplicated = data.view(-1, 1)[kept_inputs].expand(-1, bits_per_input)
        binarized = np.flip((duplicated >= kept_thresholds).numpy(), axis=(0,1)).flatten().astype(bool)
        encoded_dset += np.packbits(binarized).tobytes()
        encoded_dset += np.array(label).astype("u1").tobytes()
    return encoded_dset

def save_model(model, fname):
    submodel_info = []
    submodel_info.append({
        "num_filters": model.wisard_models[0].num_filters,
        "num_filter_inputs": model.wisard_models[0].unit_inputs,
        "num_filter_entries": model.wisard_models[0].unit_entries,
        "num_filter_hashes": model.wisard_models[0].unit_hashes,
        "num_null_bits":model.wisard_models[0].pad,
        "nonsparse_filter_idxs": [sorted(list(d.nonsparse_filters.keys())) for d in model.wisard_models[0].discriminators],
        "input_order": list(model.wisard_models[0].input_order),
        "hash_values": next(iter(model.wisard_models[0].discriminators[0].nonsparse_filters.values())).hash_values
    })
    model_info = {
        "num_inputs": model.num_inputs * model.bits_per_input,
        "num_classes": len(model.wisard_models[0].discriminators),
        "bits_per_input": model.bits_per_input,
        "bias": list(model.bias),
        "submodel_info": submodel_info
    }
    state_dict = {
        "info": model_info,
        "model": model
    }

    with lzma.open(fname, "wb") as f:
        pickle.dump(state_dict, f)

def run_inference(model_fname, dset_fname):
    with lzma.open(model_fname, "rb") as f:
        model = pickle.load(f)["model"]
    num_inputs = model.num_inputs
    total = 0
    correct = 0
    with open(dset_fname, "rb") as ds:
        bytes_per_sample = int.from_bytes(ds.read(4), "big")
        ds.read(1) # Unused
        bits_per_input = int.from_bytes(ds.read(1), "big")
        while True:
            sample_bytes = ds.read(bytes_per_sample)
            if len(sample_bytes) == 0:
                break # EOF
            sample_bits = np.unpackbits(np.frombuffer(sample_bytes, dtype="u1"))
            trimmed_sample_bits = sample_bits[0:bits_per_input*num_inputs]
            sample = np.flip(trimmed_sample_bits.reshape(-1, bits_per_input), axis=(0,1)).flatten().astype(bool)
            label = int.from_bytes(ds.read(1), "big")
            prediction = model.predict(sample)
            if prediction == label:
                correct += 1
            total += 1
            if total % 1000 == 0:
                print(total)
    print(f"Correct: {correct}/{total} ({(100*correct)/total}%)")

def main(model_fname):
    model = torch.load(model_fname).to("cpu")
    _, test_dataset = main_backprop.get_dataset()
    print("Get unused inputs")
    unused_inputs, input_remap = get_unused_inputs(model)
    print("Finalize model")
    finalized_model = finalize_model(model, unused_inputs, input_remap)
    print("Convert dataset")
    thresholds = model.thresholds
    encoded_dataset = thermometer_encode_dataset(test_dataset, thresholds, unused_inputs)
    model_out_fname = os.path.splitext(model_fname)[0] + "_finalized.pickle.lzma"
    print("Save results")
    save_model(finalized_model, model_out_fname)
    dset_out_fname = f"UNSWNB15_encoded_{thresholds.shape[1]}b.bds"
    with open(dset_out_fname, "wb") as f:
        f.write(encoded_dataset)
    print("Run inference (debug)")
    run_inference(model_out_fname, dset_out_fname)

if __name__ == "__main__":
    model_fname = sys.argv[1]
    main(model_fname)

