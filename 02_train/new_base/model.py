import torch
import unet


class UNet(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up):

        super().__init__()

        self.unet = unet.UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                constant_upsample=True)

        self.mask_head = unet.ConvPass(num_fmaps, 1, [[1, 1]], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)
        mask = self.mask_head(z)

        return mask


class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target):#, weights):

        #scale = (weights * (pred - target) ** 2)
        scale = (pred - target) ** 2

        #if len(torch.nonzero(scale)) != 0:

        #    mask = torch.masked_select(scale, torch.gt(weights, 0))
        #    loss = torch.mean(mask)

        #else:

        #    loss = torch.mean(scale)
        loss = torch.mean(scale)

        return loss

    def forward(
            self,
            affs_prediction,
            affs_target):
            #affs_weights):

        loss = self._calc_loss(affs_prediction, affs_target)#, affs_weights)

        return loss
