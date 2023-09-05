import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from base_network import MySequential, LayerNorm2d


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chan=3, dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_chan, dim, kernel_size=to_2tuple(3), padding=1, bias=bias)
        self.ipa = IPA(dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.ipa(x)
        return x


# Global Context-aware Attention (GCA)
class GCA(nn.Module):
    def __init__(self, dim, reduction=8, bias=False):
        super().__init__()

        self.context_mask = nn.Conv2d(dim, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, bias=bias)
        )

    def get_context(self, x):
        input_ = x
        b, c, h, w = x.shape
        input_ = input_.view(b, c, h * w)
        input_ = input_.unsqueeze(1)
        context_mask = self.context_mask(x)
        context_mask = context_mask.view(b, 1, h * w)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.unsqueeze(3)
        context = torch.matmul(input_, context_mask)
        context = context.view(b, c, 1, 1)
        return context

    def forward(self, x):
        context = self.get_context(x)
        context = self.mlp(context)
        x = x + context
        return x


# Intra-view Physics-aware Attention (IPA)
class IPA(nn.Module):
    def __init__(self, dim, reduction=8, bias=False):
        super(IPA, self).__init__()
        hidden_features = int(dim // reduction)

        self.ap = nn.AdaptiveAvgPool2d(1)
        self.a = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=to_2tuple(1), bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features, dim, kernel_size=to_2tuple(1), bias=bias),
            nn.Sigmoid()
        )
        self.t = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias),
            nn.Conv2d(dim, hidden_features, kernel_size=to_2tuple(1), bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features, dim, kernel_size=to_2tuple(1), bias=bias),
            nn.Sigmoid()
        )
        self.s = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        t = self.t(x)
        a = self.a(self.ap(x))
        s = self.s(x)
        x = x * t - s + (1-t) * a
        return x


# Context-aware Dimension-wise Queried Block (CDQB)
class CDQB(nn.Module):
    def __init__(self, dim, x=8, y=8, bias=False):
        super().__init__()

        partial_dim = int(dim // 4)

        self.hw = nn.Parameter(torch.ones(1, partial_dim, x, y), requires_grad=True)
        self.conv_hw = nn.Conv2d(partial_dim, partial_dim, kernel_size=to_2tuple(3), padding=1, groups=partial_dim, bias=bias)

        self.ch = nn.Parameter(torch.ones(1, 1, partial_dim, x), requires_grad=True)
        self.conv_ch = nn.Conv1d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.cw = nn.Parameter(torch.ones(1, 1, partial_dim, y), requires_grad=True)
        self.conv_cw = nn.Conv1d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.partial_gca = GCA(partial_dim)

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias),
        )

    def forward(self, x):
        input_ = x
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # hw
        x1 = x1 * self.conv_hw(F.interpolate(self.hw, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ch
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2 * self.conv_ch(
            F.interpolate(self.ch, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # cw
        x3 = x3.permute(0, 2, 1, 3)
        x3 = x3 * self.conv_cw(
            F.interpolate(self.cw, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)

        x4 = self.partial_gca(x4)
        x = self.mlp(torch.cat([x1, x2, x3, x4], dim=1)) + input_
        return x


# Cross-view Multi-dimension Interacting Attention
class CMIA(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.scale_h = nn.Parameter(torch.ones((1)), requires_grad=True)
        self.scale_w = nn.Parameter(torch.ones((1)), requires_grad=True)

        self.norm_l = LayerNorm2d(dim)
        self.norm_r = LayerNorm2d(dim)
        self.pconv_q_l = nn.Conv2d(dim, dim, to_2tuple(1), bias=bias)
        self.pconv_q_r = nn.Conv2d(dim, dim, to_2tuple(1), bias=bias)

        self.pconv_v_l = nn.Conv2d(dim, dim, to_2tuple(1), bias=bias)
        self.pconv_v_r = nn.Conv2d(dim, dim, to_2tuple(1), bias=bias)

        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim // 3, dim // 24, to_2tuple(1), bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 24, dim * 2 // 3, to_2tuple(1), bias=bias)
        )

        self.layer_scale_1 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x_l, x_r):

        q_l_h, q_l_w, q_l_c = self.pconv_q_l(self.norm_l(x_l)).chunk(3, dim=1)
        q_r_h, q_r_w, q_r_c = self.pconv_q_r(self.norm_r(x_r)).chunk(3, dim=1)

        v_l_h, v_l_w, v_l_c = self.pconv_v_l(x_l).chunk(3, dim=1)
        v_r_h, v_r_w, v_r_c = self.pconv_v_r(x_r).chunk(3, dim=1)

        # h
        q_l_h = q_l_h.permute(0, 3, 2, 1)
        q_r_h = q_r_h.permute(0, 3, 1, 2)
        v_l_h = v_l_h.permute(0, 3, 2, 1)
        v_r_h = v_r_h.permute(0, 3, 2, 1)
        attn_h = torch.matmul(q_l_h, q_r_h) * self.scale_h
        attn_h = torch.softmax(attn_h, dim=-1)
        x_r2l_h = torch.matmul(attn_h, v_r_h)
        x_l2r_h = torch.matmul(attn_h.transpose(-1, -2), v_l_h)

        # w
        q_l_w = q_l_w.permute(0, 2, 3, 1)
        q_r_w = q_r_w.permute(0, 2, 1, 3)
        v_l_w = v_l_w.permute(0, 2, 3, 1)
        v_r_w = v_r_w.permute(0, 2, 3, 1)
        attn_w = torch.matmul(q_l_w, q_r_w) * self.scale_w
        x_r2l_w = torch.matmul(torch.softmax(attn_w, dim=-1), v_r_w)
        x_l2r_w = torch.matmul(torch.softmax(attn_w.transpose(-1, -2), dim=-1), v_l_w)

        # c
        b, c, h, w = q_l_c.shape
        feats = torch.cat([q_l_c, q_r_c], dim=1)
        feats = feats.view(b, 2, c, h, w)

        attn_c = self.mlp(self.ap(torch.sum(feats, dim=1)))
        attn_l_c, attn_r_c = torch.softmax(attn_c.view(b, 2 * c, 1, 1), dim=1).chunk(2, dim=1)
        x_r2l_c = attn_l_c * v_l_c
        x_l2r_c = attn_r_c * v_r_c

        x_r2l = torch.cat([x_r2l_h.permute(0, 3, 2, 1), x_r2l_w.permute(0, 3, 1, 2), x_r2l_c], dim=1)
        x_l2r = torch.cat([x_l2r_h.permute(0, 3, 2, 1), x_l2r_w.permute(0, 3, 1, 2), x_l2r_c], dim=1)

        x_l = x_r2l * self.layer_scale_1 + x_l
        x_r = x_l2r * self.layer_scale_2 + x_r

        return x_l, x_r


class BasicBlock(nn.Module):
    def __init__(self, dim, cv=True, bias=False):
        super().__init__()
        self.cross_view = cv

        self.cdqb = CDQB(dim, bias=bias)
        self.cv = CMIA(dim) if self.cross_view else None

    def forward(self, *feats):
        feats = tuple([self.cdqb(x) for x in feats])

        if self.cross_view:
            feats = self.cv(*feats)
        return feats


class MQINet(nn.Module):
    def __init__(self, in_chan=3, dim=48, num_blks=128, bias=False):
        super().__init__()
        self.conv_in = OverlapPatchEmbed(in_chan=in_chan, dim=dim, bias=bias)
        self.body = MySequential(*[BasicBlock(dim, cv=i<num_blks-1, bias=bias) for i in range(num_blks)])
        self.conv_out = nn.Conv2d(dim, in_chan, kernel_size=to_2tuple(3), padding=1, bias=bias)

    def forward(self, x):
        input_ = x
        feats = x.chunk(2, dim=1)
        feats = [self.conv_in(i) for i in feats]
        feats = self.body(*feats)
        x = torch.cat([self.conv_out(j) for j in feats], dim=1)
        x = x + input_
        return x


if __name__ == '__main__':

    x = torch.randn((1, 6, 64, 64)).cuda()

    net = MQINet().cuda()

    y = net(x)
    print(y.shape)

    from thop import profile, clever_format

    flops, params = profile(net, (x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)





