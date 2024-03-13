import math
import os
import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import ModelAvg,get_far_set,get_close_set


def train(model, args):
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()

    ################################### Distance based vector embedding ####################
    A_blocks = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                            requires_grad=False).float()  # Look up table for blocks
    Embed = torch.zeros(args.clas, args.batchSize, args.l, args.clas)  # 2**m, bs, l, 2**m default(8, 8192, 17, 8)
    for i in range(args.clas):  # arg.clas = 2**m
        embed = torch.zeros(args.clas)
        for j in range(args.clas):  #normalize vector embedding
            if args.embed_normalize == True:
                embed[j] = (torch.sum(torch.abs(A_blocks[i, :] - A_blocks[j, :])) - 3 / 2) / 0.92582  # normalize embedding
            else:
                embed[j] = torch.sum(torch.abs(A_blocks[i, :] - A_blocks[j, :]))
        Embed[i, :, :, :] = embed.repeat(args.batchSize, args.l, 1)
    threshold = int((args.K / 2)+1)
    close_prob = torch.tensor([math.comb(args.K, k) * np.power(0.5, args.K) for k in range(threshold)])
    far_prob = torch.tensor([math.comb(args.K, k) * np.power(0.5, args.K) for k in range(threshold,args.K+1)])
    ################################## Start Training #######################################################
    for step in range(args.start_step, args.total_steps):
        ################################################################################################################
        # Curriculum learning
        if step < args.steps_snr1:
            snr1 = args.start_snr1 * (1 - step / args.steps_snr1) + (step / args.steps_snr1) * args.snr1
            snr2 = args.start_snr2
        elif args.steps_snr1 <= step < (args.steps_snr1 + args.steps_snr2):
            snr1 = args.snr1
            snr2 = args.start_snr2 * (1 - (step - args.steps_snr1) / args.steps_snr2) + (
                        (step - args.steps_snr1) / args.steps_snr2) * args.snr2
        else:
            snr1 = args.snr1
            snr2 = args.snr2
        # Data enhancement
        if (args.steps_snr1 + args.steps_snr2) < step <= (args.steps_snr1 + args.steps_snr2 + args.enhance_steps):
            std1 = 10 ** (-torch.normal(snr1, args.sigma_snr1, (1,),
                                        requires_grad=False).item() * 1.0 / 10 / 2)  # forward snr
            std2 = 10 ** (-torch.normal(snr2, args.sigma_snr2, (1,),
                                        requires_grad=False).item() * 1.0 / 10 / 2)  # feedback snr
        else:
            std1 = 10 ** (-snr1 * 1.0 / 10 / 2)  # forward snr
            std2 = 10 ** (-snr2 * 1.0 / 10 / 2)  # feedback snr
        # Noise values for the parity bits
        fwd_noise_par = torch.normal(0, std=std1, size=(args.batchSize, args.l, args.T), requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.l, args.T - 1, args.num_feedback),
                                    requires_grad=False)
        #extra channel for sending random sequence
        pre_fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.l, args.m),
                                    requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0 * fb_noise_par
            pre_fb_noise_par = 0 * pre_fb_noise_par
        # random generate a r length code, recevier send to transmitter
        # seq = torch.randint(0,args.clas, (args.batchSize, args.l, args.seq_length))
        # seq_binary = torch.randint(0,2,(args.batchSize, args.seq_length*args.K))
        # seq_binary = ((seq.unsqueeze(-1) & ((1<<torch.arange(3)).flip(-1))) > 0).float().contiguous().view(args.batchSize,-1)
        # Generate the bits
        bVec = torch.randint(0, args.clas, (args.batchSize, args.l, 1))
        bVec_binary = ((bVec.unsqueeze(-1) & ((1<<torch.arange(args.m)).flip(-1))) > 0).float().contiguous().view(args.batchSize,-1)
        seq_md = torch.zeros((args.batchSize, args.l, args.clas*args.seq_length),
                             requires_grad=False)
        # TODO"1.二进制+N+MLE+bulls+十进制+MD 2.MD+PC+N+MLE+bulls“ 模型的input要不要MLE
        # TODO：接受之后再做bulls检测，不合格的扔掉
        # add noise on seq
        # TODO：测试不同的threshold
        # bulls = (seq_binary == bVec_binary).sum(dim=1)
        # a = sum(bulls[:] > 24)
        # while sum(bulls[:] < threshold) > 0:
        #     seq_binary[bulls[:] < threshold, :] = torch.randint(0, 2, (sum(bulls[:] < threshold), args.K))
        #     bulls = (seq_binary == bVec_binary).sum(dim=1)
        # time2 = time.time()
        if args.train_domain == 'close':
            seq_binary = get_close_set(bVec_binary,close_prob)
        elif args.train_domain == 'far':
            seq_binary = get_far_set(bVec_binary,far_prob)
        else:
            seq_binary = torch.randint(0,2,(args.batchSize,args.K))
        #add noise on seq
        seq_binary = seq_binary.float().reshape(args.batchSize,args.l,args.m)
        seq_binary *= 2
        seq_binary -= 1
        seq_binary += pre_fb_noise_par
        # test = torch.where(seq_binary > 1.1, 1, 0).float().sum()
        #sampling decision
        seq_binary = torch.where(seq_binary > 0, 1, -1).float()
        seq = (((seq_binary[:, :, 0]+1)*2)+(seq_binary[:, :, 1]+1)+((seq_binary[:, :, 2]+1)*0.5)).unsqueeze(-1)
        for i in range(args.clas):
            mask = (seq == i).long()
            seq_md = seq_md + (mask * Embed[i, :, :, :])
        bVec_md = torch.zeros((args.batchSize, args.l, args.clas),
                                  requires_grad=False)  # generated data in terms of distance embeddings
        for i in range(args.clas):
            mask = (bVec == i).long()
            bVec_md = bVec_md + (mask * Embed[i, :, :, :])
        #mix bits and random sequence
        #bVec_md = combine_code(bVec_md, seq_md)
        # Generate noise sequence (Curriculum learning strategy)
        if np.mod(step, args.core) == 0:
            w_locals = []
            w0 = model.state_dict()
            w0 = copy.deepcopy(w0)
        else:
            # Use the common model to have a large batch strategy
            model.load_state_dict(w0)

        # feed into model to get predictions
        preds = model(step, bVec_md.cuda(), seq_md.cuda(), fwd_noise_par.cuda(), fb_noise_par.cuda(),
                      A_blocks.cuda(), isTraining=1)

        args.optimizer.zero_grad()

        ys = bVec.contiguous().view(-1)

        preds = preds.contiguous().view(-1, preds.size(-1))  # => (Batch*K) x 2
        #
        preds = torch.log(preds)
        loss = F.nll_loss(preds, ys.cuda())
        loss.backward()
        ####################### Gradient Clipping optional ###########################
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        ##############################################################################
        args.optimizer.step()
        # Save the model
        w1 = model.state_dict()
        w_locals.append(copy.deepcopy(w1))
        ###################### untill core number of iterations are completed ####################
        if np.mod(step, args.core) != args.core - 1:
            continue
        else:
            ########### When core number of models are obtained #####################
            w2 = ModelAvg(w_locals)  # Average the models
            model.load_state_dict(copy.deepcopy(w2))
            ##################### change the learning rate ##########################
            if args.use_lr_schedule:
                args.scheduler.step()
        ################################ Observe test accuracy ##############################
        with torch.no_grad():
            probs, decodeds = preds.max(dim=1)
            succRate = sum(decodeds == ys.cuda()) / len(ys)
            log = 'GBAF_new: Idx={}, lr={:.3}, snr1={:.3f} ,snr2={:.3f} ,BS={} ,loss={:.3} ,BER={:.3} ,num={}'.format(step,
                 args.optimizer.state_dict()['param_groups'][0]['lr'], snr1, snr2, args.batchSize, loss.item(), 1 - succRate.item(), sum(decodeds != ys.cuda()).item())
            print(log)
            with open(args.log_path, 'a') as f:
                f.write(log + '\n')
        ####################################################################################
        # if np.mod(step, args.core * 50) == args.core - 1:
        #     epoch_loss_record.append(loss.item())
        #     if not os.path.exists('weights'):
        #         os.mkdir('weights')
        #     torch.save(epoch_loss_record, 'weights/loss')

        # Save the model
        if np.mod(step, args.save_interval) == args.core - 1 and (step + 1) >= (args.steps_snr1 + args.steps_snr2):
            if not os.path.exists(args.saveDir):
                os.mkdir(args.saveDir)
            saveDir = args.saveDir + 'model_weights' + str(step)
            torch.save(model.state_dict(), saveDir)
    else:
        if not os.path.exists(args.saveDir):
            os.mkdir(args.saveDir)
        saveDir = args.saveDir + 'latest'
        torch.save(model.state_dict(), saveDir)
