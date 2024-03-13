from utils import *
from model import Model
from parameters import args_parser
from test import test
from train import train
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    ##################### Initialize the parameters ################
    args = args_parser()

    ##################### Initialize the model ################
    args.attention_size = args.num_heads_att * args.dim_att
    model = Model(args)

    # n_gpus = 2
    # torch.distributed.init_process_group("nccl", world_size=n_gpus)
    # torch.cuda.set_device()

    model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    ##################### Initialize the optimizer ################
    args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                       weight_decay=args.wd, amsgrad=False)
    if args.lambda_type == 0:
        lr_lambda = lambda epoch: (1 - epoch / (args.total_steps - args.start_step))
    elif args.lambda_type == 1:
        lr_lambda = lambda epoch: 1 if epoch < args.steps_snr1*2 else 0.1 if epoch < args.steps_snr1*4 else 0.01
    args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lr_lambda)

    ##################### Initialize the optimizer ################
    if args.start_step !=0:
        model.load_state_dict(torch.load(args.start_model))
    if args.train == 1:
        train(model, args)
    else:
        test(model, args)
