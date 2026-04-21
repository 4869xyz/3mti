import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.load_spec_model import Encoder, Upsample


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x):
    return x * torch.sigmoid(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Normalize(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Normalize(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Diff_ResnetBlock(nn.Module):
    def __init__(
            self,
            channels,
            out_channels,
            emb_channels=64,
            dropout=False,
            use_conv=True,
            use_scale_shift_norm=True,
            use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            Normalize(channels),
            nn.SiLU(),
            torch.nn.Conv2d(channels,
                            self.out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            Normalize(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            torch.nn.Conv2d(self.out_channels,
                            self.out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = torch.nn.Conv2d(channels,
                                                   self.out_channels,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        else:
            self.skip_connection = torch.nn.Conv2d(channels,
                                                   self.out_channels,
                                                   kernel_size=1,
                                                   stride=1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h



class AttnBlock(nn.Module):
    def __init__(self, in_channels, head=32):
        super().__init__()
        self.in_channels = in_channels
        self.head = head

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, kv):
        h_ = x
        h_, kv = self.norm(h_), self.norm(kv)
        q = self.q(h_)
        k = self.k(kv)
        v = self.v(kv)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b * self.head, c // self.head, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b * self.head, c // self.head, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b * self.head, c // self.head, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_



class Diff_Encoder(nn.Module):
    def __init__(self, in_channels, num_down, emb_channels=64, base_ch=64,
                 dropout=0.0,
                 use_conv=True,
                 use_scale_shift_norm=True,
                 use_checkpoint=False):
        super().__init__()
        assert num_down >= 1, "num_down must be >= 1"

        self.num_down = num_down
        self.deconv = DoubleConv(in_channels, 64)


        chs = [base_ch * (2 ** i) for i in range(0, num_down)]
        chs = [chs[i] if i < 2 else chs[i - 1] for i in range(len(chs))]
        self.out_channels_per_level = chs

        self.down_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        self.attnBlock = AttnBlock(chs[-2])

        for i in range(num_down - 1):
            in_ch = chs[i]
            out_ch = chs[i + 1]

            self.down_blocks.append(Down(in_ch, out_ch))

            self.res_blocks.append(
                Diff_ResnetBlock(
                    channels=out_ch,
                    out_channels=out_ch,
                    emb_channels=emb_channels,
                    dropout=dropout,
                    use_conv=use_conv,
                    use_scale_shift_norm=use_scale_shift_norm,
                    use_checkpoint=use_checkpoint,
                )
            )


    def forward(self, x, emb, inter_feat):
        feats = []
        x = self.deconv(x)
        feats.append(x)
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
            if i == len(self.down_blocks) - 2:
                x = x + inter_feat
            x = self.res_blocks[i](x, emb)

            feats.append(x)

        return feats


class Diff_Decoder(nn.Module):
    def __init__(
            self,
            chs,
            out_channels=1,
            emb_channels=64,
            dropout=0.0,
            use_conv=True,
            use_scale_shift_norm=True,
            use_checkpoint=False,
    ):
        super().__init__()
        self.num_levels = len(chs)
        self.up_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        self.level_to_res_idx = {}



        for i in range(self.num_levels - 1, 0, -1):
            dec_ch = chs[i] 
            skip_ch = chs[i - 1]  

            self.up_blocks.append(
                Upsample(in_channels=dec_ch, out_channels=skip_ch)
            )

            self.res_blocks.append(
                Diff_ResnetBlock(
                    channels=skip_ch,
                    out_channels=skip_ch,
                    emb_channels=emb_channels,
                    dropout=dropout,
                    use_conv=use_conv,
                    use_scale_shift_norm=use_scale_shift_norm,
                    use_checkpoint=use_checkpoint,
                )
            )


        self.final_upsample = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(self, feats, emb, inter_feats):
        y = feats[-1] 

        for k, i in enumerate(range(self.num_levels - 1, 0, -1)):
            skip = feats[i - 1]
            y = self.up_blocks[k](y) + skip
            if k == 0:
                y = y+ inter_feats

            y = self.res_blocks[k](y, emb)

        y = self.final_upsample(y)
        return y


class FMM_Diff(nn.Module):
    def __init__(self, cfg,n_channels=1, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.ch = cfg.model.t_emb

        self.cfg = cfg

        self.num_modality = len(cfg.data.modalities_name)
        self.model_spe_list = nn.ModuleList(
            [Encoder(cfg.model.in_channels, cfg.model.down_num) for _ in range(self.num_modality)]
        )
        self.model_map_list = nn.ModuleList(
            [Encoder(self.num_modality - 1, cfg.model.down_num) for _ in range(self.num_modality)]
        )

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.ch),
            torch.nn.Linear(self.ch,
                            self.ch),
        ])

        self.diff_encoder = Diff_Encoder(cfg.model.in_channels, cfg.model.down_num, emb_channels=cfg.model.t_emb,
                                         base_ch=cfg.model.base_ch)

        feats_ch = self.diff_encoder.out_channels_per_level

        self.diff_decoder = Diff_Decoder(chs=feats_ch, out_channels=1, emb_channels=cfg.model.t_emb)

        self.att_list = nn.ModuleList(
            [AttnBlock(feats_ch[-1]) for _ in range(self.num_modality)]
        )

        self.con_list = nn.ModuleList(
            [torch.nn.Conv2d(feats_ch[-1] * (self.num_modality - 1), feats_ch[-1], kernel_size=1, stride=1, padding=0)
             for _ in range(self.num_modality)]
        )

        self.attn_MSFM = AttnBlock(feats_ch[-1])

        self.Res_list = nn.ModuleList([
            Diff_ResnetBlock(
                channels=feats_ch[i] * len(cfg.data.modalities_name),
                out_channels=feats_ch[i],
                emb_channels=cfg.model.t_emb,
                dropout=False
            )
            for i in range(-1,-3,-1)
        ])



    def forward(self, x, condition, t):
        temb = get_timestep_embedding(t, self.cfg.model.t_emb)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        MFSM_input, inter_feature = self.get_condition(condition)

        MFSM_output = self.MSFM(MFSM_input)

        att_f_list = []
        for i in range(len(MFSM_output)):
            kv = torch.cat(MFSM_output[ :i] + MFSM_output[i + 1:], dim=1)
            kv = self.con_list[i](kv)
            att_f_list.append(self.att_list[i](MFSM_output[i],kv))

        b_feat = torch.cat(att_f_list, dim=1)
        b_feat = self.Res_list[0](b_feat,temb)

        inter_feature = self.MSFM(inter_feature)
        inter_feature = torch.cat(inter_feature, dim=1)
        inter_feature = self.Res_list[1](inter_feature,temb)
        feats = self.diff_encoder(x, temb,inter_feature)

        b_feat = self.attn_MSFM(feats[-1], b_feat)

        N = len(feats)
        out_feats = [feats[i] for i in range(N - 1)] + [b_feat]
        out = self.diff_decoder(out_feats, temb, inter_feature)

        return out

    def get_condition(self, input_list):
        inter_feature = []
        MFSM_input = []

        for i in range(self.num_modality):
            sep_output = self.model_spe_list[i](input_list[:,i:i+1])
            map_input = torch.cat([input_list[:, :i], input_list[:, i + 1:]], dim=1)
            map_output = self.model_map_list[i](map_input)
            MFSM_input.append(sep_output[-1] + map_output[-1])
            inter_feature.append(sep_output[-2] + map_output[-2])

        return MFSM_input,inter_feature

    def MSFM(self, MFSM_input):
        gap = nn.AdaptiveAvgPool2d(1)
        MFSM_out = [torch.sigmoid(gap(x)) for x in MFSM_input]

        MFSM_cat = torch.cat(MFSM_out, dim=3)

        MFSM_soft = torch.softmax(MFSM_cat, dim=3)
        MFSM_weight = torch.split(MFSM_soft, split_size_or_sections=1, dim=3)

        MFSM_weight = list(MFSM_weight)

        return [(x * w) + x for x, w in zip(MFSM_input, MFSM_weight)]

    def load_encoders(self, ):
        for index_model in range(self.num_modality):
            name_of_modality = self.cfg.data.modalities_name[index_model].split('.')[0]
            spe_encoder = self.model_spe_list[index_model]
            map_encoder = self.model_map_list[index_model]
            self.load_parameter(spe_encoder, name_of_modality, spec=True)
            self.load_parameter(map_encoder, name_of_modality, spec=False)

    def load_parameter(self, encoder, name_of_modality, spec=True):
        if spec:
            dir = "specific_encoder"
        else:
            dir = "mapping_encoder"
        decoder_path = os.path.join(self.cfg.train.ckp_point_path, name_of_modality, dir,
                                    f"model_epoch{self.cfg.train.decoder_ckp_for_load}.pth")
        ckp_decoder = torch.load(decoder_path, map_location='cpu')

        encoder.load_state_dict(ckp_decoder['model_state_dict'])
        logging.info('successfully loading {} for {}'.format(dir, name_of_modality))

        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

