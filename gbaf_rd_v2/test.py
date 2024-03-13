import math
import time

import numpy as np
import torch
import pdb
from GBAF.model import GBAF
from utils import get_far_set,get_close_set

def test(model, args):
    print("-->-->-->-->-->-->-->-->-->--> start testing ...")
    checkpoint = torch.load(args.save_path)
    gbaf = GBAF(args).cuda()
    gbaf_cp = torch.load(args.gbaf)
    # # ======================================================= load weights
    model.load_state_dict(checkpoint)
    model.eval()
    gbaf.load_state_dict(gbaf_cp)
    gbaf.eval()
    map_vec = 2 ** (torch.arange(args.m))
    threshold = int((args.K / 2) + 1)
    close_prob = torch.tensor([math.comb(args.K, k) * np.power(0.5, args.K) for k in range(threshold)])
    far_prob = torch.tensor([math.comb(args.K, k) * np.power(0.5, args.K) for k in range(threshold, args.K + 1)])

    ################################### Distance based vector embedding ####################
    A_blocks = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                            requires_grad=False).float()  # Look up table for blocks
    Embed = torch.zeros(args.clas, args.batchSize, args.l, args.clas)
    for i in range(args.clas):
        embed = torch.zeros(args.clas)
        for j in range(args.clas):
            if args.embed_normalize == True:
                embed[j] = (torch.sum(torch.abs(A_blocks[i, :] - A_blocks[j, :])) - 3 / 2) / 0.866
            else:
                embed[j] = torch.sum(torch.abs(A_blocks[i, :] - A_blocks[j, :]))
        Embed[i, :, :, :] = embed.repeat(args.batchSize, args.l, 1)
    Table = Embed.unique()
    table_repeat = Table.repeat(args.batchSize, args.l).reshape(args.batchSize, args.l, Table.shape[0]).unsqueeze(-2)\
        .expand(args.batchSize, args.l,args.clas*args.seq_length, Table.shape[0])
    # failbits = torch.zeros(args.K).cuda()
    bitErrors = 0
    pktErrors = 0
    print('SNR1=',args.snr1,'\tSNR2=',args.snr2)
    # generate n sequence
    std1 = 10 ** (-args.snr1 * 1.0 / 10 / 2)
    std2 = 10 ** (-args.snr2 * 1.0 / 10 / 2)
    fwd_noise_par = torch.normal(0, std=std1, size=(args.batchSize, args.l, args.T), requires_grad=False)
    fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.l, args.T - 1, args.num_feedback),
                                requires_grad=False)
    pre_fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.l, args.m),
                                    requires_grad=False)
    if args.snr2 == 100:
        fb_noise_par = 0 * fb_noise_par
        pre_fb_noise_par = 0 * pre_fb_noise_par
    for step in range(args.test_steps):
        # Generate the bits
        bVec = torch.randint(0, args.clas, (args.batchSize, args.l, 1))
        #attention test
        # bVec[42, :, :] = torch.tensor([0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0])\
        #     .reshape(args.l, 1)
        ##################
        bVec_binary = ((bVec.unsqueeze(-1) & ((1 << torch.arange(3)).flip(-1))) > 0).float().contiguous().view(
            args.batchSize, -1)
        # TODO:测试bulls要加完噪声然后用MLE还原回来再求
        seq_md = torch.zeros((args.batchSize, args.l, args.clas * args.seq_length),
                             requires_grad=False)
        seq_binary = torch.randint(0, 2, (args.batchSize, args.K))
        bulls = (seq_binary == bVec_binary).sum(dim=1)
        threshold = (args.l * 3 / 2)
        if args.train_domain == 'close':
            seq_binary = get_close_set(bVec_binary,close_prob)
        elif args.train_domain == 'far':
            seq_binary = get_far_set(bVec_binary,far_prob)
        else:
            pass
        #domain C
        # while sum(bulls[:] < threshold) > 0:
        #     seq_binary[bulls[:] < threshold, :] = torch.randint(0, 2, (sum(bulls[:] < threshold), args.K))
        #     bulls = (seq_binary == bVec_binary).sum(dim=1)
        #domain F
        # while sum(bulls[:] >= threshold) > 0:
        #     seq_binary[bulls[:] >= threshold, :] = torch.randint(0, 2, (sum(bulls[:] >= threshold), args.K))
        #     bulls = (seq_binary == bVec_binary).sum(dim=1)
        # add noise on seq
        seq_binary = seq_binary.reshape(args.batchSize, args.l, args.m)
        seq_binary = torch.where(seq_binary == 0, -torch.ones_like(seq_binary), seq_binary).float() \
                     + pre_fb_noise_par
        seq_binary = torch.where(seq_binary > 0, 1, -1).float()
        seq = (torch.where(seq_binary==1,1,0)[:,:,0]*4+
               torch.where(seq_binary==1,1,0)[:,:,1]*2+
                torch.where(seq_binary==1,1,0)[:,:,2]*1).float().reshape(args.batchSize, args.l, 1)
        #attention test
        # seq[42, :, :] = torch.tensor([0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0]) \
        #     .reshape(args.l, 1)
        ##################
        for i in range(args.clas):
            mask = (seq == i).long()
            seq_md = seq_md + (mask * Embed[i, :, :, :])
        # start = time.time()
        # seq_md_n = seq_md + pre_fb_noise_par
        # seq_md_n = Table[torch.argmin(torch.abs(seq_md_n.unsqueeze(-1) - table_repeat), dim=-1)]
        # end = time.time()
        # print("end:",end-start)
        bVec_md = torch.zeros((args.batchSize, args.l, args.clas),
                              requires_grad=False)  # generated data in terms of distance embeddings
        for i in range(args.clas):
            mask = (bVec == i).long()
            bVec_md = bVec_md + (mask * Embed[i, :, :, :])

        # feed into model to get predictions
        with torch.no_grad():
            preds = torch.zeros((args.batchSize, args.l, args.clas)).cuda()
            preds_a = gbaf(step, bVec_md.cuda(),fwd_noise_par.cuda(),
                          fb_noise_par.cuda(), A_blocks.cuda(), isTraining=0)
            preds_b = model(step, bVec_md.cuda(),seq_md.cuda(), fwd_noise_par.cuda(),
                           fb_noise_par.cuda(), A_blocks.cuda(), isTraining=0)
            #C2F1
            # preds[(bulls[:] < threshold),:,:] = preds_a[(bulls[:] < threshold),:,:]
            # preds[(bulls[:] >= threshold),:,:] = preds_b[(bulls[:] >= threshold),:,:]
            #C1F2
            # preds[(bulls[:] >= threshold),:,:] = preds_a[(bulls[:] >= threshold),:,:]
            # preds[(bulls[:] < threshold),:,:] = preds_b[(bulls[:] < threshold),:,:]
            preds = preds_b
            ys = bVec.contiguous().view(-1)
            preds = preds.contiguous().view(-1, preds.size(-1))
            probs, decodeds = preds.max(dim=1)
            decisions = decodeds != ys.cuda()
            bitErrors += decisions.sum()

            BER = bitErrors / (step + 1) / args.batchSize / args.l
            pktErrors += decisions.view(args.batchSize, args.l).sum(1).count_nonzero()
            PER = pktErrors / (step + 1) / args.batchSize
            print('GBAF_FESv1', 'num, BER, errors, PER, errors = ', step, round(BER.item(), 10), bitErrors.item(),
                  round(PER.item(), 10), pktErrors.item(), )

    BER = bitErrors.cpu() / (args.test_steps * args.batchSize * args.l)
    PER = pktErrors.cpu() / (args.test_steps * args.batchSize)
    print("Final test BER = ", torch.mean(BER).item())
    # pdb.set_trace()

