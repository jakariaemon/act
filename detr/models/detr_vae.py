# detr_vae.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable

from .backbone import build_backbone, AudioBackbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """
    DETR-based model that also does a VAE-style latent encoding of an action sequence,
    and can fuse image + audio + state inputs.
    """
    def __init__(self,
                 backbones,
                 transformer,
                 encoder,
                 state_dim,
                 num_queries,
                 camera_names,
                 audio_backbone=None,  # NEW
                 use_audio=False       # NEW
    ):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.use_audio = use_audio
        self.audio_backbone = audio_backbone

        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # If we have image backbones
        if backbones is not None and len(backbones) > 0:
            self.backbones = nn.ModuleList(backbones)
            # We assume all image backbones have same num_channels, so just use the first
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # fallback in case no image
            self.backbones = None
            self.input_proj = None
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)

        # AUDIO PROJECTION
        if self.use_audio and self.audio_backbone is not None:
            # Project audio feature maps to the same hidden dim
            self.audio_input_proj = nn.Conv2d(self.audio_backbone.num_channels, hidden_dim, kernel_size=1)
        else:
            self.audio_input_proj = None

        # ------------------ VAE part ------------------
        self.latent_dim = 32  # final size of latent z
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim)  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)   # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to [mu, logvar]
        self.register_buffer('pos_table',
                             get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))  # for [CLS], qpos, action_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # learned position embedding for proprio + latent tokens


    def forward(self, qpos, image, env_state, actions=None, is_pad=None, audio=None):
        """
        qpos: (B, 14)
        image: (B, num_cams, C=3, H, W) or None
        env_state: not used in your current code, but we keep param
        actions: (B, seq, 14) or None
        is_pad: (B, seq) or None
        audio: (B, 1, Freq, Time) or None
        """
        device = qpos.device
        bs = qpos.shape[0]
        is_training = (actions is not None)

        ### 1) VAE encoder step
        if is_training:
            # project action + qpos to embedding dim, concat CLS token
            action_embed = self.encoder_action_proj(actions)  # (B, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)  # (B,1,hidden_dim)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # (B,1,hidden_dim)

            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)  # (B, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, B, hidden_dim)

            # build mask if is_pad is given
            if is_pad is not None:
                cls_joint_is_pad = torch.full((bs, 2), False, device=device)
                is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)  # (B, seq+1)

            # position embedding for encoder
            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)  # (seq+1, 1, hidden_dim)

            # run encoder
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_cls = encoder_output[0]  # (B, hidden_dim), the first token (CLS)

            # get latent
            latent_info = self.latent_proj(encoder_cls)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)  # (B, hidden_dim)
        else:
            # no actions => no training => use zeros
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=device)
            latent_input = self.latent_out_proj(latent_sample)

        ### 2) Construct the Transformer "src" from images and/or audio
        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            # gather image features from each camera
            for cam_id, cam_name in enumerate(self.camera_names):
                feats, poses = self.backbones[cam_id](image[:, cam_id])  # each returns ([feat], [pos])
                # feats[0] is the final feature map
                proj_feat = self.input_proj(feats[0])  # (B, hidden_dim, H, W)
                all_cam_features.append(proj_feat)
                all_cam_pos.append(poses[0])  # might be None if your backbone returns None

            # Concat camera features along width dimension
            src = torch.cat(all_cam_features, dim=3)  # e.g. (B, hidden_dim, H, sum_of_W)
            # If you have position embeddings for each camera, you could cat them as well
            if all_cam_pos[0] is not None:
                pos = torch.cat([p for p in all_cam_pos if p is not None], dim=3)
            else:
                pos = None

            # If using audio
            if self.use_audio and self.audio_backbone is not None and audio is not None:
                audio_feats, audio_pos = self.audio_backbone(audio)  # returns ([feat], [pos])
                audio_proj = self.audio_input_proj(audio_feats[0])   # (B, hidden_dim, H_a, W_a)
                # For simplicity, concat to src on width
                src = torch.cat([src, audio_proj], dim=3)
                # If there's an audio pos embedding, concat likewise
                if audio_pos[0] is not None and pos is not None:
                    pos = torch.cat([pos, audio_pos[0]], dim=3)
                else:
                    # else pos remains None or partial
                    pass
        else:
            # fallback if no image backbone
            src = None
            pos = None

        # Proprio input
        if self.backbones is not None:
            # e.g. just project qpos
            proprio_input = self.input_proj_robot_state(qpos)
        else:
            # fallback from your original code
            qpos_ = self.input_proj_robot_state(qpos)
            env_ = self.input_proj_env_state(env_state)
            src = torch.cat([qpos_, env_], dim=1)
            proprio_input = None

        ### 3) Pass to Transformer decoder
        if src is not None:
            hs = self.transformer(
                src,  # (B, hidden_dim, H, W)
                mask=None,
                query_embed=self.query_embed.weight,
                pos_embed=pos if pos is not None else torch.zeros_like(src),
                latent_input=latent_input,     # (B, hidden_dim)
                proprio_input=proprio_input,   # (B, hidden_dim)
                additional_pos_embed=self.additional_pos_embed.weight
            )[0]  # shape (1, B, num_queries, hidden_dim)
        else:
            # If no vision/audio, e.g. state-only fallback
            # You might do something different here or skip the Transformer entirely.
            # For now, just define an empty/hacky approach:
            hs = torch.zeros((1, bs, self.num_queries, self.transformer.d_model),
                             device=device)

        # 4) Final MLP heads
        a_hat = self.action_head(hs)     # shape (1, B, num_queries, state_dim)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):
    """
    Your existing CNN + MLP model. Unchanged, except we imported AudioBackbone if needed.
    """
    def __init__(self, backbones, state_dim, camera_names):
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        bs, _ = qpos.shape
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)
        features = torch.cat([flattened_features, qpos], axis=1)
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim  # e.g. 256
    dropout = args.dropout     # e.g. 0.1
    nhead = args.nheads        # e.g. 8
    dim_feedforward = args.dim_feedforward  # e.g. 2048
    num_encoder_layers = args.enc_layers     # e.g. 4
    normalize_before = args.pre_norm         # e.g. False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    return encoder


def build(args):
    """
    Build the DETRVAE model, optionally with audio if args.use_audio == True.
    """
    state_dim = 14

    # Build image backbones
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    # Optionally build audio backbone
    audio_backbone = None
    if getattr(args, "use_audio", False):
        audio_backbone = AudioBackbone()

    transformer = build_transformer(args)
    encoder = build_encoder(args)

    model = DETRVAE(
        backbones=backbones,
        transformer=transformer,
        encoder=encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        audio_backbone=audio_backbone,    # pass in audio backbone
        use_audio=args.use_audio          # control usage
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_cnnmlp(args):
    """
    Build the simpler CNN + MLP model, ignoring audio.
    """
    state_dim = 14
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones=backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    return model