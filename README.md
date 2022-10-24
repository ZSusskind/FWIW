# FWIW
[![DOI](https://zenodo.org/badge/556513783.svg)](https://zenodo.org/badge/latestdoi/556513783)

Code to accompany the paper:

**An FPGA-Based Weightless Neural Network for Edge Network
Intrusion Detection**, Zachary Susskind, Aman Arora, Alan T. L. Bacellar, Diego L. C. Dutra, Igor D. S. Miranda, Mauricio Breternitz Jr., Priscila M. V. Lima, Felipe M. G. França, and Lizy K. John

*Under Submission*

# Usage
## Prerequisites

Our codebase was written for Python 3.8.10; other version may very well work but are untested.

We recommend constructing a virtual environment for dependency management:
```
python3 -m venv env
source env/bin/activate
```

From here, dependency installation can be automatically handled with a single command:
```
pip install -r requirements.txt
```

Training models and generating RTL does not require any proprietary software. However, running the functional testbench requires a VCS installation and a valid license.
Set the `VCS_HOME` environment variable to the top-level VCS installation directory so that the executable path is `$(VCS_HOME)/bin/vcs`.
Power and area estimates are from Vivado and require a licensed installation; however, we also provide the reports for reference.

## Training the model
All relevant code lives in the `software_model/` directory. The training flow is in three parts: creating the model, pruning the model, and converting the PyTorch model to a binary form which is consumed by the RTL generator. To train a model with the same hyperparameters used in the paper, run:

```
./main_backprop.py
./prune_model.py unswnb15.pt 0.3
./finalize_model.py unswnb15_pruned_0_3.pt
```

This produces the output `unswnb15_pruned_0_3_finalized.pickle.lzma`.
Some minor run-to-run variation in accuracy may occur. We eliminate unused inputs from the model and converted dataset; the set of unused inputs may also vary.

To view model coefficients and plot the ROC, run `./get_coefficients.py unswnb15_pruned_0_3.pt`.

## Producing RTL
All relevant code lives in the `rtl/` directory.

We provide a Makefile for generating the RTL. Invoking `make` with no arguments will generate RTL for a sample model (our own pretrained model), and then attempt to build the RTL and testbenches. To generate the RTL without building it, run `make template`.  
The Makefile also allows for the model file, data bus width, and target throughput to be specified as optional command-line arguments. So for instance, you could run:
```
make template MODEL=../software_model/foobar.pickle.lzma BUS_WIDTH=32 XPUT=16
```

SystemVerilog sources are generated using the [Mako templating library](https://www.makotemplates.org/) from the `.sv.mako` sources under `rtl/mako_srcs/`, and written under `rtl/sv_srcs/`.

## RTL Functional Correctness
We provide a full-system testbench to check the functional correctness of the RTL model. Building and running this testbench requires VCS. To build and run with the default (pretrained) options:
```
cd rtl/
make testbenches
cd testbench_objs/
./test_functional_correctness +DSET=../../software_model/pretrained_encoded_4b.bds
```
Note the specification of the testbench as a `.bds` file. `finalize_model.py` produces this file as one of its outputs.

## RTL Power and Area
Replicating RTL power/energy/area results requires a Vivado license. Our numbers are derived from post-implementation reports.  
For runs targeting the larger (`xcvu9p-flgb2104-2-i`) FPGA, we run in out-of-context mode to analyse the model as a component in a larger design.
For runs targeting the smaller (`xc7s6csga225-2`) FPGA (with 64b and 80b bus widths), we do *not* run in out-of-context model.  
All synthesis runs were performed using the `Flow_PerfOptimized_high` synthesis strategy in order to aggressively minimize cycle time.
We used Vivado 2019.2, but do not anticipate that other versions would give significantly different results.


