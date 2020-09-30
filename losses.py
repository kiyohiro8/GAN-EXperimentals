import torch

class HingeLoss():
    def __init__(self, batch_size, device, unet=False):

        self.ones = torch.ones(1).to(device)
        self.zeros = torch.zeros(1).to(device)

    def gen_loss(self, logit):
        return - torch.mean(logit)

    def dis_real_loss(self, real_logit):
        return - torch.mean(torch.min(real_logit - 1, self.zeros)) 
        #return - torch.mean(torch.relu(- real_logit + 1)) - torch.mean(torch.relu(fake_logit + 1)) 

    def dis_fake_loss(self, fake_logit):
        return - torch.mean(torch.min(- fake_logit - 1, self.zeros))

    def dis_cutmix_loss(self, cutmix_logit, mask):
        loss = torch.min(cutmix_logit - 1, self.zeros) * mask + \
               torch.min(- cutmix_logit - 1, self.zeros) * (1 - mask)
        return - torch.mean(loss)

class WGAN_GP():
    
    def __init__(self, dis, drift=0.001, use_gp=True):
        self.dis = dis
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps,
                           reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param height: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samps + ((1 - epsilon) * fake_samps)
        merged.requires_grad_(True)

        # forward pass
        op = self.dis(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=torch.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps)
        real_out = self.dis(real_samps)

        loss = (torch.mean(fake_out) - torch.mean(real_out)
                + (self.drift * torch.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps):
        # calculate the WGAN loss for generator
        loss = -torch.mean(self.dis(fake_samps))

        return loss