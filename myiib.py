import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def _kl_div(r,A, lambda_, mean_r, std_r):
    """ Return the feature-wise KL-divergence of p(z|x) and q(z)
        # The equation in the paper is:
        # Z = λ * R + (1 - λ) * ε)
        # where ε ~ N(μ_r, σ_r**2),
        #  and given R the distribution of Z ~ N(λ * R, ((1 - λ)*σ_r)**2) (λ * R is constant variable)
        #
        # As the KL-Divergence stays the same when both distributions are scaled,
        # normalizing Z such that Q'(Z) = N(0, 1) by using σ(R), μ(R).
        # Then for Gaussian Distribution:
        #   I(R, z) = KL[P(z|R)||Q(z)] = KL[P'(z|R)||N(0, 1)]
        #           = 0.5 * ( - log[det(noise)] - k + tr(noise_cov) + μ^T·μ )
    """

    v_z = (1+2*(lambda_)**2-2*lambda_)
    R_sub_A_std = ((r - A)/std_r)**2
    log_v_z = torch.log(v_z)

    # print(v_z.mean())

    my_capacity = -0.5*(log_v_z + 1 - R_sub_A_std - v_z)

    # r_norm = (r - mean_r) / std_r
    # var_z = (1 - lambda_) ** 2
    # log_var_z = torch.log(var_z)
    # mu_z = r_norm * lambda_

    # capacity = -0.5 * (1 + log_var_z - mu_z ** 2 - var_z)
    return my_capacity


class SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels, used to smoothen the input """
    def __init__(self, kernel_size, sigma, channels):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "kernel_size must be an odd number (for padding), {} given".format(self.kernel_size)
        variance = sigma ** 2.
        x_cord = torch.arange(kernel_size, dtype=torch.float)  # 1, 2, 3, 4
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)  # 1, 2, 3 \ 1, 2, 3 \ 1, 2, 3
        y_grid = x_grid.t()  # 1, 1, 1 \ 2, 2, 2 \ 3, 3, 3
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_3d = kernel_2d.expand(channels, 1, -1, -1)  # expand in channel dimension
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels,
                              padding=0, kernel_size=kernel_size,
                              groups=channels, bias=False)
        self.conv.weight.data.copy_(kernel_3d)
        self.conv.weight.requires_grad = False
        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))

    def forward(self, x):
        return self.conv(self.pad(x))



class iib(nn.Module):
    def __init__(self):
        super(iib, self).__init__()
        # self.N = batchsize
        self.conv1 = nn.Conv2d(512, 512//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512//2, 512*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512*2, 512, kernel_size=3, padding=1)
        self.sigmod = nn.Sigmoid()
        self.smooth = SpatialGaussianKernel(kernel_size = 1,sigma=1*0.25,channels=512)
        self._alpha_bound = 5

        self.decode = nn.Sequential(nn.Conv2d(512,512,kernel_size=7,stride=2),nn.BatchNorm2d(512),nn.ReLU(),nn.Conv2d(512,512,kernel_size=7,stride=2),
                                    nn.BatchNorm2d(512),nn.ReLU(),nn.Conv2d(512,512,kernel_size=3))
        self.adaptivemaxpool2d = nn.AdaptiveMaxPool2d((1, 1))
        # self.device = device
    def forward(self,featureout,batchsize):
        alpha = F.relu(self.conv1(featureout))
        alpha = F.relu(self.conv2(alpha))
        alpha = self.conv3(alpha)
        alpha = alpha.clamp(-self._alpha_bound, self._alpha_bound)
        lambda_ = self.sigmod(alpha)
        lambda_ = self.smooth(lambda_)
        # eps_out1 = torch.randn(featureout.shape,device=torch.device("cuda:0"))
        # eps_out2 = torch.randn(featureout.shape,device=torch.device("cuda:0"))
        # for i in range(batchsize):
        #
        #     R = featureout[i,...]
        #     m_r = torch.mean(R,dim=0)
        #     std_r = torch.std(R,dim=0)
        #     eps1 = torch.randn(size=R.shape).cuda() * std_r + m_r
        #     _,m,n = eps1.shape
        #     eps1 = torch.reshape(eps1,(1,512,m,n))
        #     eps_out1[i,...] = eps1
        #
        #     eps2 = torch.randn(size=R.shape).cuda() * std_r + m_r
        #     eps2 = torch.reshape(eps2, (1, 512, m, n))
        #     eps_out2[i, ...] = eps2

        b, _, m, n = featureout.shape
        m_r = torch.reshape(torch.mean(featureout, dim=1), (b, 1, m, n))
        std_r = torch.reshape(torch.std(featureout, dim=1), (b, 1, m, n))
        eps_out1 = torch.randn(size=featureout.shape, device=torch.device("cuda")) * std_r + m_r
        eps_out2 = torch.randn(size=featureout.shape, device=torch.device("cuda")) * std_r + m_r

        Z = featureout*lambda_ + (1. - lambda_) * eps_out1
        A = featureout*(1. - lambda_) + lambda_ * eps_out2
        info = _kl_div(featureout,A,lambda_,m_r,std_r)
        info = info.mean(dim=0)
        # print(info)
        info = info.mean()
        Z_id = self.decode(Z)
        # print(Z_id.shape)
        Z_id = self.adaptivemaxpool2d(Z_id)
        # print(Z_id.shape)
        Z_id = Z_id.view(-1,512)
        # print(info.item())
        return Z,A,lambda_,info,Z_id,eps_out1,eps_out2


if __name__ == '__main__':
    net = iib().cuda()
    feature = torch.randn(4,512,28,28).cuda()
    out,A,_,info,Z_id,_,_ = net(feature,4)
    print(out.shape)
    print(A.shape)
    print(info.item())
    print(Z_id.shape)
