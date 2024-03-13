import os
import torch
import torch.nn as nn
from utils import PositionalEncoder, Power_reallocate, Power_reallocate_fb
from model_components import Encoder, Decoder


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.pe = PositionalEncoder()

        # Transmitter
        self.Tmodel = Encoder(args.clas*(1+args.seq_length) + (1 + args.num_feedback) * (args.T - 1), args.m, args.attention_size, args.num_layers_T,
                              args.num_heads_att, args.dropout)
        # Feedback
        if args.active_feedback == True:
            self.Fmodel = Encoder(args.clas*args.seq_length+args.T - 1 + args.num_feedback * (args.T - 2), args.m, args.attention_size,
                                  args.num_layers_F, args.num_heads_att, args.dropout, args.num_feedback)
        # Receiver
        self.Rmodel = Decoder(args.clas*args.seq_length+args.T, args.m, args.attention_size, args.num_layers_R, args.num_heads_att,
                              args.dropout, args.dropout_output)

        #Power Reallocation as in deepcode work
        if self.args.reloc == 1:
            self.total_power_reloc1 = Power_reallocate(args)
            self.total_power_reloc2 = Power_reallocate_fb(args)

    # TODO:1.use the average parameters of train
    def power_constraint_T(self, inputs, isTraining, step, idx=0):
        if isTraining == 1:
            # train
            this_mean = torch.mean(inputs, 0)
            this_std = torch.std(inputs, 0)
        elif isTraining == 0:
            # test
            if not os.path.exists('statistics'):
                os.mkdir('statistics')
            # assume the batch size is fixed
            if step == 0:
                this_mean = torch.mean(inputs, 0)
                this_std = torch.std(inputs, 0)

                torch.save(this_mean, 'statistics/this_mean_T' + str(idx))
                torch.save(this_std, 'statistics/this_std_T' + str(idx))
            elif step <= 100:
                this_mean = (torch.load('statistics/this_mean_T' + str(idx)) + torch.mean(inputs, 0)) / 2
                this_std = (torch.load('statistics/this_std_T' + str(idx)) + torch.std(inputs, 0)) / 2

                torch.save(this_mean, 'statistics/this_mean_T' + str(idx))
                torch.save(this_std, 'statistics/this_std_T' + str(idx))
            else:
                this_mean = torch.load('statistics/this_mean_T' + str(idx))
                this_std = torch.load('statistics/this_std_T' + str(idx))

        outputs = (inputs - this_mean) * 1.0 / (this_std + 1e-8)
        return outputs


    def power_constraint_F(self, inputs, isTraining, step, idx=0):
        if isTraining == 1:
            # train
            this_mean = torch.mean(inputs, 0)
            this_std = torch.std(inputs, 0)
        elif isTraining == 0:
            # test
            if not os.path.exists('statistics'):
                os.mkdir('statistics')
            # assume the batch size is fixed
            if step == 0:
                this_mean = torch.mean(inputs, 0)
                this_std = torch.std(inputs, 0)

                torch.save(this_mean, 'statistics/this_mean_F' + str(idx))
                torch.save(this_std, 'statistics/this_std_F' + str(idx))
            elif step <= 100 :
                this_mean = (torch.load('statistics/this_mean_F' + str(idx)) + torch.mean(inputs, 0)) / 2
                this_std = (torch.load('statistics/this_std_F' + str(idx)) + torch.std(inputs, 0)) / 2

                torch.save(this_mean, 'statistics/this_mean_F' + str(idx))
                torch.save(this_std, 'statistics/this_std_F' + str(idx))
            else:
                this_mean = torch.load('statistics/this_mean_F' + str(idx))
                this_std = torch.load('statistics/this_std_F' + str(idx))

        outputs = (inputs - this_mean) * 1.0 / (this_std + 1e-8)
        return outputs


    def forward(self, step, bVec_md,seq_md,fwd_noise_par, fb_noise_par, table=None, isTraining=1):
        # combined_noise_par = fwd_noise_par + fb_noise_par
        combined_noise_par = None
        #random generate a r length code, recevier send to transmitter
        #noise_random_sequence = random_sequence + fb_noise_rans[:,:,:]
        for t in range(self.args.T):
            ################### Initialize input of Tmodel #############################################################
            if t == 0:
                x_T = torch.cat([bVec_md,seq_md, torch.zeros(self.args.batchSize, self.args.l, (1 + self.args.num_feedback) * (self.args.T - 1)).cuda()], dim=2)
                output_T = self.Tmodel(x_T, None, self.pe)
                parity = self.power_constraint_T(output_T, isTraining, step, t)
                if self.args.reloc == 1:
                    parity = self.total_power_reloc1(parity, t)

                parity_all = parity
                received = parity + fwd_noise_par[:, :, t].unsqueeze(-1)
                # Generate feedback
                if self.args.active_feedback == False:
                    feedback = self.power_constraint_F(received, isTraining, step, t) + fb_noise_par[:, :, t, :]
                else:  # active feedback
                    #concat the random code
                    x_F = torch.cat([seq_md,received, torch.zeros(self.args.batchSize, self.args.l, self.args.num_feedback * (self.args.T - 2) + self.args.T -2).cuda()], dim=2)
                    output_F = self.Fmodel(x_F, None, self.pe)
                    output_F = self.power_constraint_F(output_F, isTraining, step, t)
                    if self.args.reloc == 1:
                        output_F = self.total_power_reloc2(output_F, t)
                    parity_fb = output_F
                    feedback = output_F + fb_noise_par[:, :, t, :]


            elif t == self.args.T-2:
                x_T = torch.cat([bVec_md,seq_md, parity_all,
                                 torch.zeros(self.args.batchSize, self.args.l, self.args.T - (t + 1)).cuda(),
                                 feedback,
                                 torch.zeros(self.args.batchSize, self.args.l, self.args.num_feedback * (self.args.T - (t + 1))).cuda()], dim=2)
                output_T = self.Tmodel(x_T, None, self.pe)
                parity = self.power_constraint_T(output_T, isTraining, step, t)
                if self.args.reloc == 1:
                    parity = self.total_power_reloc1(parity, t)

                parity_all = torch.cat([parity_all, parity], dim=2)

                received = torch.cat([received, parity + fwd_noise_par[:, :, t].unsqueeze(-1)], dim=2)
                # Generate feedback
                if self.args.active_feedback == False:
                    feedback = self.power_constraint_F(received, isTraining, step, t) + fb_noise_par[:, :, t, :]
                else:  # active feedback
                    x_F = torch.cat([seq_md,received, parity_fb], dim=2)
                    output_F = self.Fmodel(x_F, None, self.pe)
                    output_F = self.power_constraint_F(output_F, isTraining, step, t)
                    if self.args.reloc == 1:
                        output_F = self.total_power_reloc2(output_F, t)
                    parity_fb = torch.cat([parity_fb, output_F],dim=2)
                    feedback = torch.cat([feedback, output_F + fb_noise_par[:, :, t, :]], dim=2)


            elif t == self.args.T-1:
                x_T = torch.cat([bVec_md,seq_md, parity_all, feedback], dim=2)
                output_T = self.Tmodel(x_T, None, self.pe)
                parity = self.power_constraint_T(output_T, isTraining, step, t)
                if self.args.reloc == 1:
                    parity = self.total_power_reloc1(parity, t)

                parity_all = torch.cat([parity_all, parity], dim=2)
                received = torch.cat([received, parity + fwd_noise_par[:, :, t].unsqueeze(-1)], dim=2)




            else:
                x_T = torch.cat([bVec_md,seq_md, parity_all,
                                 torch.zeros(self.args.batchSize, self.args.l, self.args.T - (t + 1)).cuda(), feedback,
                                 torch.zeros(self.args.batchSize, self.args.l, self.args.num_feedback * (self.args.T - (t + 1))).cuda()], dim=2)
                output_T = self.Tmodel(x_T, None, self.pe)
                parity = self.power_constraint_T(output_T, isTraining, step, t)
                if self.args.reloc == 1:
                    parity = self.total_power_reloc1(parity, t)

                parity_all = torch.cat([parity_all, parity], dim=2)
                received = torch.cat([received, parity + fwd_noise_par[:, :, t].unsqueeze(-1)], dim=2)

                # Generate feedback
                if self.args.active_feedback == False:
                    feedback = self.power_constraint_F(received, isTraining, step, t) + fb_noise_par[:, :, t, :]
                else:  # active feedback
                    x_F = torch.cat([seq_md,received,
                                     torch.zeros(self.args.batchSize, self.args.l, self.args.T - (t + 2)).cuda(),
                                     parity_fb,
                                     torch.zeros(self.args.batchSize, self.args.l, self.args.num_feedback * (self.args.T - (t + 2))).cuda()],dim=2)
                    output_F = self.Fmodel(x_F, None, self.pe)
                    output_F = self.power_constraint_F(output_F, isTraining, step, t)

                    if self.args.reloc == 1:
                        output_F = self.total_power_reloc2(output_F, t)
                    parity_fb = torch.cat([parity_fb, output_F], dim=2)  # update collected parities
                    feedback = torch.cat([feedback, output_F + fb_noise_par[:, :, t, :]], dim=2)


        ######################## Decode ###############################

        # print('\tfb:', parity_fb.std(0).sum(), parity_fb.shape)
        x_R = torch.cat([seq_md,received],dim=2)
        output = self.Rmodel(x_R, None, self.pe)
        return output


