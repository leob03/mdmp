# This code is modified from https://github.com/GuyTevet/motion-diffusion-model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from model.gcn import GraphConvolution
from data_loaders.humanml.scripts.motion_process import recover_from_ric



class MDMP(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 use_gcn=True, clip_version=None, lv=False, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset
        self.lv = lv

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.input_feats_joints = 66  # 22 joints * 3 features

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        # self.cond_mode = kargs.get('no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 1.)
        self.use_gcn = use_gcn  # whether to use GCN for input and output processing

        if not self.use_gcn:
            self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
        else:
            self.input_process = InputProcess_wGCN(self.data_rep, self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        if not self.use_gcn:
            self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats, self.lv)
        else:
            self.output_process = OutputProcess_wGCN(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats, self.lv)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # clip.model.convert_weights(
        #     clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', True)
        if 'text' in self.cond_mode:
            # enc_text = self.encode_text(y['text'])        #with the next 4 lines, we allow the model to only call CLIP once per denoising process and reuse the embeddings
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        #Incorporate motion input
        if 'motion_embed' in y.keys() and 'motion_embed_mask' in y.keys():
            # print('motion_embed' in y.keys(), "multi-modal input detected")
            motion_embed, motion_embed_mask = y['motion_embed'], y['motion_embed_mask']
            assert x.shape == motion_embed.shape == motion_embed_mask.shape
            x = (x * ~motion_embed_mask) + (motion_embed * motion_embed_mask)

        x = self.input_process(x) # [bs, seqlen, d]

        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError

class InputProcess_wGCN(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = GraphConvolution(self.input_feats, self.latent_dim, node_n=196, bias=True)
        if self.data_rep == 'rot_vel':
            # self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)
            raise ValueError # Not implemented yet
        

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((0, 3, 1, 2)).reshape(bs, nframes, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x.permute((1, 0, 2))
        # elif self.data_rep == 'rot_vel':
        #     first_pose = x[[0]]  # [1, bs, 150]
        #     first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
        #     vel = x[1:]  # [seqlen-1, bs, 150]
        #     vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
        #     return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats, lv):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.lv = lv
        if self.lv:
            self.poseFinal = nn.Linear(self.latent_dim, self.input_feats * 2) # Updated
        else:
            self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]
            first_pose = self.poseFinal(first_pose)
            vel = output[1:]
            vel = self.velFinal(vel)
            output = torch.cat((first_pose, vel), axis=0)
        else:
            raise ValueError
        if self.lv:
            output = output.reshape(nframes, bs, 2 * self.input_feats, self.nfeats)  # Updated
        else:
            output = output.reshape(nframes, bs, self.input_feats, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, 2 *njoints, nfeats, nframes]
        return output

class OutputProcess_wGCN(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats, lv):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.lv = lv
        # Update: Output layer now has 2 * input_feats to include variance
        if self.lv:
            self.poseFinal = GraphConvolution(self.latent_dim, self.input_feats*2, node_n=196, bias=True)
        else:
            self.poseFinal = GraphConvolution(self.latent_dim, self.input_feats, node_n=196, bias=True)
        if self.data_rep == 'rot_vel':
            # self.velFinal = nn.Linear(self.latent_dim, self.input_feats)
            raise ValueError # Not implemented yet

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = output.permute(1, 0, 2)  # [bs, seqlen, d]
            output = self.poseFinal(output)
            output = output.permute(1, 0, 2)  # [seqlen, bs, d]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]
        #     first_pose = self.poseFinal(first_pose)
        #     vel = output[1:]
        #     vel = self.velFinal(vel)
        #     output = torch.cat((first_pose, vel), axis=0)
        else:
            raise ValueError
        # Updated: Reshape to include doubled features for mean and variance
        if self.lv:
            # print('output shape', output.shape) # [seqlen, bs, 132]
            output = output.reshape(nframes, bs, 2 * self.input_feats, self.nfeats)  # Updated
        else:
            output = output.reshape(nframes, bs, self.input_feats, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, 2 *njoints, nfeats, nframes] Updated
        return output

class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output