# Short initial run of the full 8-layer, width-512 model on this Mac.
# Uses the verified real-data subset; this is not a full-corpus reproduction.
out_dir = 'out-baseline-pilot'
dataset = 'smoke_real'
n_layer = 8
n_head = 8
n_embd = 512
block_size = 1023
batch_size = 2
gradient_accumulation_steps = 4
max_iters = 100
warmup_iters = 10
lr_decay_iters = 100
learning_rate = 3e-4
min_lr = 3e-5
eval_interval = 25
eval_iters = 4
log_interval = 10
compile = False
device = 'mps'
dtype = 'float32'
num_threads = 4
