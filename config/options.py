import argparse

parser = argparse.ArgumentParser(description='MARK')

parser.add_argument('--exp_name', type=str, default='Dataset_exp')
parser.add_argument('--dataset', type=str, default='None')
parser.add_argument('--clip_name', type=str, default='OpenAICLIP', help="clip scope in { OpenAICLIP, MetaCLIP, SigLIP }")

# --------------------
# DataLoader Options
# --------------------

parser.add_argument('--data_dir', type=str, default='/path/to/your/dataset_dir/') 
parser.add_argument('--data_split_dir', type=str, default='/path/to/your/dataset/split_dir')
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--data_split', type=float, default=-1.0)
parser.add_argument('--num_threads', type=int, default=1000)
parser.add_argument('--distance_metric', type=str, default='cosine', help='{cosine, hamming}')
# ----------------------
# Training Params
# ----------------------

parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=192)
parser.add_argument('--workers', type=int, default=0)
# ----------------------

opts = parser.parse_args()
opts.dataset = opts.exp_name.split('_')[0]
