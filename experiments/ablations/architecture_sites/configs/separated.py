from experiments.ablations.architecture_sites.configs.common import COMMON
globals().update(COMMON)
# A: L1 -> T -> L2 -> D -> L3:6 -> depth -> L7 -> temporal -> L8.
n_buffer = 1
n_core = 4
n_source = 1
out_dir = 'experiments/ablations/architecture_sites/results/separated'
