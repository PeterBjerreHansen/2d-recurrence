from experiments.ablations.architecture_sites.configs.common import COMMON
globals().update(COMMON)
# B: L1 -> T -> D -> L2:7 -> depth and temporal -> L8.
n_buffer = 0
n_core = 6
n_source = 0
out_dir = 'experiments/ablations/architecture_sites/results/coincident'
