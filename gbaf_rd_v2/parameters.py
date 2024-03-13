import argparse


def args_parser():
    parser = argparse.ArgumentParser()


    # Sequence arguments
    parser.add_argument('--snr1', type=float, default=2, help="Transmission SNR")
    parser.add_argument('--snr2', type=float, default=5., help="Feedback SNR")
    parser.add_argument('--K', type=int, default=48, help="Sequence length")
    parser.add_argument('--m', type=int, default=3, help="Block size")
    parser.add_argument('--l', type=int, default=16, help="Number of bit blocks")
    parser.add_argument('--T', type=int, default=9, help="Number of interactions")
    parser.add_argument('--seq_reloc', type=int, default=1)
    parser.add_argument('--memory', type=int, default=48)
    parser.add_argument('--core', type=int, default=1)

    # Transformer arguments
    parser.add_argument('--num_heads_att', type=int, default=1, help="number of heads for the multi-head attention")
    parser.add_argument('--dim_att', type=int, default=32, help="number of features for each head")
    parser.add_argument('--attention_size', type=int, default=32, help="heads_att * dim_att")
    parser.add_argument('--dropout', type=float, default=0.0, help="prob of dropout")
    parser.add_argument('--dropout_output', type=float, default=0.03, help="prob of dropout")
    parser.add_argument('--vv', type=int, default=1)
    parser.add_argument('--num_layers_R', type=int, default=3, help="number of attention layers in receiver")
    parser.add_argument('--num_layers_T', type=int, default=2, help="number of attention layers in transmitter")
    parser.add_argument('--num_layers_F', type=int, default=2, help="number of attention layers in feedback network")
    parser.add_argument('--active_feedback', type=bool, default=False, help='active feedback or negative feedback')
    parser.add_argument('--gbaf_active_feedback', type=bool, default=True, help='active feedback or negative feedback')
    parser.add_argument('--num_feedback', type=int, default=1, help='Times of the original feedback')

    # Learning arguments
    parser.add_argument('--train', type=int, default=1, help='0 for test; 1 for train')
    parser.add_argument('--start_step', type=int, default=0, help='the start step for retrained model; if not 0, start model is needed')
    parser.add_argument('--start_model', type=str, default='None', help='the path of retrained model')

    parser.add_argument('--saveDir', type=str, default='weights/', help='Save folder of the model weights')
    parser.add_argument('--gbaf', type=str, default='GBAF/2_5', help='Save path of gbaf')
    parser.add_argument('--save_path', type=str, default='weights/RD_2_5_2', help='Save path of the model weights')
    parser.add_argument('--save_interval', type=int, default=10000, help='the interval between save')
    parser.add_argument('--log_path', type=str, default='log.txt')
    parser.add_argument('--reloc', type=int, default=1, help="w/ or w/o power rellocation")
    parser.add_argument('--total_steps', type=int, default=120000, help="number of total steps to train")
    parser.add_argument('--test_steps', type=int, default=10000, help="number of total steps to test")
    parser.add_argument('--batchSize', type=int, default=4096, help="batch size of one step")
    parser.add_argument('--train_domain', type=str, default='close', help="close, far and random")


    parser.add_argument('--clip_th', type=float, default=0.5, help="clipping threshold")
    parser.add_argument('--use_lr_schedule', type=bool, default=True, help="lr scheduling")
    parser.add_argument('--embed_normalize', type=bool, default=True, help="normalize embedding")
    parser.add_argument('--clas', type=int, default=8, help="number of possible class for a block of bits, equal to 2**m")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--wd', type=float, default=0.01, help="weight decay")
    parser.add_argument('--device', type=str, default='cuda:0', help="GPU")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--lambda_type", type=int, default=0, help="0:linear, 1:step")

    # Curriculum learning arguments
    parser.add_argument('--start_snr1', type=float, default=3)
    parser.add_argument('--start_snr2', type=float, default=100)
    parser.add_argument('--steps_snr1', type=int, default=20000, help="number of snr1 curriculum learning steps to train")
    parser.add_argument('--steps_snr2', type=int, default=20000, help="number of snr2 curriculum learning steps to train")


    # Data enhancement arguments
    parser.add_argument('--enhance_steps', type=int, default=50000, help='enhance_steps (after curriculum learning)')
    parser.add_argument('--sigma_snr1', type=float, default=2, help='snr1_used = Normal(snr1, sigma_snr1)')
    parser.add_argument('--sigma_snr2', type=float, default=2, help='snr2_used = Normal(snr2, sigma_snr2)')

    #Recevier dominant random sequence
    parser.add_argument('--seq_length', type=int, default=1)

    args = parser.parse_args()

    return args
