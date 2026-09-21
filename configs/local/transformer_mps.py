from pathlib import Path
exec(Path('configs/local/recurrent_mps.py').read_text())
architecture = 'baseline'
out_dir = 'experiments/smoke/results/transformer_mps'
update_support = []
update_probabilities = []
eval_u_t = 0
eval_u_d = 0
