import argparse
parser = argparse.ArgumentParser(description = 'Argument Parser.')

# folder settings
parser.add_argument('--output_dir', help='output_dir', type=str)
parser.add_argument('--data_root', help='data_root', type=str)
parser.add_argument('--temp_output_dir', help='temp storage for philly data io', default='', type=str)

# env settings
parser.add_argument('--gpu_id', help='gpu_list eg 0,1,2,3', type=str)

# analysis settings
parser.add_argument('--profile', help='enable profile 0/1', default=1, type=int)
parser.add_argument('--enable_summary', help='enable_summary', default=1, type=int)
parser.add_argument('--enable_validation', help='enable_validation', default=1, type=int)

# experiment settings for chart gan
parser.add_argument('--dataset', help='dataset', type=str)
parser.add_argument('--exp_type', help='exp_type', type=str)
parser.add_argument('--phase_kimg', help='phase_kimg for progressive', default=1, type=int)
parser.add_argument('--data_shuffle', help='use data_shuffle', default=1, type=int)
parser.add_argument('--batch_group', help='set group num per batch', type=int)
# parser.add_argument('--batch_size', help='set batch size per batch for ae', default=-1, type=int)
parser.add_argument('--smooth_loss', help='smooth_loss', default=0.0, type=float)
parser.add_argument('--syn_batch', help='syn_batch', type=int)
parser.add_argument('--real_batch', help='real_batch', type=int)
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--grad_repeat', help='grad_repeat to enlarge batchsize', default=1, type=int)
parser.add_argument('--grad_weight', help='normal_weight', default=0.0, type=float)
parser.add_argument('--random_bg', help='using random background', default=1, type=int)
parser.add_argument('--random_shift', help='using random shift', default=1, type=int)

# experiment settings for style gan
parser.add_argument('--stylegan_batch_size', help='set stylegan_batch_size', default=-1, type=int)
parser.add_argument('--apply_noise', help='apply_noise for tex generator', default=True, type=bool)

# previous model
parser.add_argument('--model_path', help='previous network model', default='', type=str)


