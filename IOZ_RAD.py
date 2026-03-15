from typing import Union, List, Optional
import numpy as np
import torch
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
from .clip_model import CLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from sklearn.cluster import KMeans


class GradFlipOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale_factor):
        ctx.scale_factor = scale_factor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.scale_factor, None

class GradFlipModule(nn.Module):
    def __init__(self):
        super(GradFlipModule, self).__init__()

    def apply(self, x, scale_factor):
        return GradFlipOp.apply(x, scale_factor)


class DomainClassifierNet(nn.Module):
    def __init__(self, feat_dim, mid_dim):
        super(DomainClassifierNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim // 2),
            nn.ReLU(),
            nn.Linear(mid_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)

class ProjectionBranch(nn.Module):
    def __init__(self, in_features, out_features, n_parallel=1, parallel_mode=False,
                 normalize=False, act_fn='relu', drop_ratio=0.0):
        super(ProjectionBranch, self).__init__()

        self.parallel_mode = parallel_mode
        self.n_parallel = n_parallel
        self.normalize = normalize

        if act_fn == 'relu':
            self.act = nn.ReLU()
        elif act_fn == 'gelu':
            self.act = nn.GELU()
        elif act_fn == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1)
        elif act_fn == 'none':
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU()

        self.drop = nn.Dropout(drop_ratio) if drop_ratio > 0 else nn.Identity()

        if parallel_mode:
            branch_dim = out_features // n_parallel
            self.branch_layers = nn.ModuleList([
                nn.Linear(in_features, branch_dim) for _ in range(n_parallel)
            ])

            self.output_proj = nn.Linear(branch_dim * n_parallel, out_features)

            if normalize:
                self.norm_layers = nn.ModuleList([
                    nn.LayerNorm(branch_dim) for _ in range(n_parallel)
                ])
            else:
                self.norm_layers = None
        else:
            self.fc = nn.Linear(in_features, out_features)

            if normalize:
                self.norm_layer = nn.LayerNorm(out_features)
            else:
                self.norm_layer = None

    def forward(self, x):
        if self.parallel_mode:
            branch_outs = []

            for h in range(self.n_parallel):
                branch_val = self.branch_layers[h](x)
                branch_val = self.act(branch_val)

                if self.normalize and self.norm_layers is not None:
                    branch_val = self.norm_layers[h](branch_val)

                branch_val = self.drop(branch_val)
                branch_outs.append(branch_val)

            fused_out = torch.cat(branch_outs, dim=-1)
            result = self.output_proj(fused_out)
        else:
            result = self.fc(x)
            result = self.act(result)

            if self.normalize and self.norm_layer is not None:
                result = self.norm_layer(result)

            result = self.drop(result)

        return result

class FeatureMapper(nn.Module):
    def __init__(self, in_features, out_features, n_copies, do_stack=False, array_input=True,
                 parallel_mode=False, n_parallel=4, act_fn='relu', normalize=False,
                 drop_ratio=0.0):
        super(FeatureMapper, self).__init__()

        self.n_copies = n_copies
        self.do_stack = do_stack
        self.array_input = array_input
        self.parallel_mode = parallel_mode
        self.n_parallel = n_parallel

        self.branches = nn.ModuleList([
            ProjectionBranch(
                in_features,
                out_features,
                n_parallel=n_parallel,
                parallel_mode=parallel_mode,
                normalize=normalize,
                act_fn=act_fn,
                drop_ratio=drop_ratio
            ) for _ in range(n_copies)
        ])

    def forward(self, tokens):
        result_tokens = []

        for i in range(self.n_copies):
            if self.array_input:
                mapped = self.branches[i](tokens[i][:, 1:, :])
            else:
                mapped = self.branches[i](tokens)

            result_tokens.append(mapped)

        if self.do_stack:
            result_tokens = torch.stack(result_tokens, dim=1)

        return result_tokens

class ContextInjectionModule(nn.Module):
    def __init__(self, feat_dim, ctx_len, inject_depth, text_mode, ctx_type, active=True):
        super(ContextInjectionModule, self).__init__()

        self.feat_dim = feat_dim
        self.ctx_len = ctx_len
        self.inject_depth = inject_depth
        self.text_mode = text_mode
        self.active = active

        self.ctx_type = ctx_type

        if self.active:
            if 'S' in ctx_type:
                self.learnable_ctx = nn.ParameterList(
                    [nn.Parameter(torch.empty(self.ctx_len, self.feat_dim))
                     for _ in range(self.inject_depth)])

                for param in self.learnable_ctx:
                    nn.init.normal_(param, std=0.02)

            if 'D' in ctx_type:
                self.generated_ctx = [0.]

    def update_generated_ctx(self, generated_ctx):
        self.generated_ctx = generated_ctx

    def inject_text(self, block, idx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.active:
            ctx_len = self.ctx_len

            if idx < self.inject_depth:
                if 'S' in self.ctx_type and 'D' in self.ctx_type:
                    fixed_ctx = self.learnable_ctx[idx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    text_ctx = self.generated_ctx + fixed_ctx
                elif 'S' in self.ctx_type:
                    fixed_ctx = self.learnable_ctx[idx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    text_ctx = fixed_ctx
                elif 'D' in self.ctx_type:
                    text_ctx = self.generated_ctx
                else:
                    print('At least one context type must be selected when injection branches are enabled.')
                    raise NotImplementedError

            if idx == 0:
                x = x
            else:
                if idx < self.inject_depth:
                    head_part = x[:1, :, :]
                    tail_part = x[1 + ctx_len:, :, :]
                    text_ctx = text_ctx.permute(1, 0, 2).half()
                    x = torch.cat([head_part, text_ctx, tail_part], dim=0)
                else:
                    x = x
        else:
            x = x

        x, attn_out = block(q_x=x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)

        return x, attn_out

    def inject_visual(self, block, idx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.active:
            ctx_len = self.ctx_len

            if idx < self.inject_depth:
                if 'S' in self.ctx_type and 'D' in self.ctx_type:
                    fixed_ctx = self.learnable_ctx[idx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    vis_ctx = self.generated_ctx + fixed_ctx
                elif 'S' in self.ctx_type:
                    fixed_ctx = self.learnable_ctx[idx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    vis_ctx = fixed_ctx
                elif 'D' in self.ctx_type:
                    vis_ctx = self.generated_ctx
                else:
                    print('At least one context type must be selected when injection branches are enabled.')
                    raise NotImplementedError


            if idx == 0:
                vis_ctx = vis_ctx.permute(1, 0, 2).half()
                x = torch.cat([x, vis_ctx], dim=0)
            else:
                if idx < self.inject_depth:
                    kept = x[0:x.shape[0] - ctx_len, :, :]
                    vis_ctx = vis_ctx.permute(1, 0, 2).half()
                    x = torch.cat([kept, vis_ctx], dim=0)
                else:
                    x = x
        else:
            x = x

        x, attn_out = block(q_x=x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)

        if self.active:
            tokens = x[:x.shape[0] - ctx_len, :, :]
        else:
            tokens = x

        return x, tokens, attn_out

    def forward(self, block, idx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.text_mode:
            return self.inject_text(block, idx, x, k_x, v_x, attn_mask)
        else:
            return self.inject_visual(block, idx, x, k_x, v_x, attn_mask)


class LinguisticEncoderModule(nn.Module):
    def __init__(self, frozen):
        super(LinguisticEncoderModule, self).__init__()
        self.word_encoder = _Tokenizer()
        self.cached_text_repr = {}
        self.normal_templates = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw',
                              '{} without defect',
                              '{} without damage']
        self.anomaly_templates = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        self.template_groups = [self.normal_templates, self.anomaly_templates]
        self.sentence_wrappers = ['a bad photo of a {}.',
                                 'a low resolution photo of the {}.',
                                 'a bad photo of the {}.',
                                 'a cropped photo of the {}.',
                                 ]
        self.frozen = frozen

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
        torch.IntTensor, torch.LongTensor]:
        if isinstance(texts, str):
            texts = [texts]

        bos_id = self.word_encoder.encoder["<|startoftext|>"]
        eos_id = self.word_encoder.encoder["<|endoftext|>"]
        token_sequences = [[bos_id] + self.word_encoder.encode(text) + [eos_id] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(token_sequences), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(token_sequences), context_length, dtype=torch.int)

        for i, tokens in enumerate(token_sequences):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eos_id
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def forward(self, model, texts, device, vis_conditioning_batch: Optional[torch.Tensor] = None):
        text_repr_list = []

        for t_idx, text in enumerate(texts):
            vis_cond_single = None
            if vis_conditioning_batch is not None:
                if t_idx < vis_conditioning_batch.shape[0]:
                    vis_cond_single = vis_conditioning_batch[t_idx:t_idx+1]
                else:
                    pass

            if self.frozen:
                if self.cached_text_repr.get(text) is None:
                    lang_features = self.build_text_repr(model, text, device, vis_conditioning_single=vis_cond_single)
                    self.cached_text_repr[text] = lang_features
                else:
                    lang_features = self.cached_text_repr[text]
            else:
                lang_features = self.build_text_repr(model, text, device, vis_conditioning_single=vis_cond_single)
                self.cached_text_repr[text] = lang_features


            text_repr_list.append(lang_features)

        lang_features = torch.stack(text_repr_list, dim=0)
        lang_features = F.normalize(lang_features, dim=1)

        return lang_features

    def build_text_repr(self, model, text, device, vis_conditioning_single: Optional[torch.Tensor] = None):
        lang_features = []
        for i in range(len(self.template_groups)):
            raw_text = text
            raw_text = raw_text.replace('-', ' ')
            described_states = [state.format(raw_text) for state in self.template_groups[i]]
            full_sentences = []
            for s in described_states:
                for wrapper in self.sentence_wrappers:
                    full_sentences.append(wrapper.format(s))

            tokenized_sents = self.tokenize(full_sentences, context_length=77).to(device)

            n_variations = tokenized_sents.shape[0]
            broadcast_vis_cond = None
            if vis_conditioning_single is not None:
                broadcast_vis_cond = vis_conditioning_single.expand(n_variations, -1)

            category_embeds = model.extract_linguistic(tokenized_sents, vis_conditioning=broadcast_vis_cond)

            category_embeds /= category_embeds.norm(dim=-1, keepdim=True)
            category_embed = category_embeds.mean(dim=0)
            category_embed /= category_embed.norm()
            lang_features.append(category_embed)

        lang_features = torch.stack(lang_features, dim=1)

        return lang_features


class MultiScaleFeatureAggregation(nn.Module):
    def __init__(self, n_groups):
        super(MultiScaleFeatureAggregation, self).__init__()
        self.n_groups = n_groups
        self.n_candidate_tokens = n_groups * 5
        self.grouping_algo = KMeans(n_clusters=self.n_groups, n_init=10, max_iter=300, random_state=42)
        self.min_groups = 1

    def forward(self, spatial_tokens: list, score_maps: list):
        score_map = torch.mean(torch.stack(score_maps, dim=1), dim=1)
        score_map = torch.softmax(score_map, dim=2)[:, :, 1]

        salient_tokens = []
        k = min(score_map.shape[1], self.n_candidate_tokens)
        topk_pos = torch.topk(score_map, k=k, dim=1).indices
        for layer in range(len(spatial_tokens)):
            picked_tokens = spatial_tokens[layer]. \
                gather(dim=1, index=topk_pos.unsqueeze(-1).
                       expand(-1, -1, spatial_tokens[layer].shape[-1]))
            salient_tokens.append(picked_tokens)

        merged_features = torch.cat(salient_tokens, dim=2)

        batch_centroids = []
        for b in range(merged_features.shape[0]):
            grouping_input = merged_features[b, :, :].detach().cpu().numpy()

            if not np.isfinite(grouping_input).all():
                print(f"Warning: Batch {b} contains non-finite values, cleaning data...")
                grouping_input = np.nan_to_num(grouping_input, nan=0.0, posinf=1.0, neginf=-1.0)

                if np.std(grouping_input) < 1e-6:
                    print(f"Warning: Batch {b} has very low variance, using mean as centroid")
                    centroid = torch.mean(merged_features[b], dim=0)
                    batch_centroids.append(centroid)
                    continue

            try:
                group_assignments = self.grouping_algo.fit_predict(grouping_input)

                centroids = []

                for gid in range(self.n_groups):
                    group_members = []
                    for token_set in salient_tokens:
                        member_feats = token_set[b, :, :][group_assignments == gid]
                        if len(member_feats) > 0:
                            group_members.append(member_feats)

                    if len(group_members) > 0:
                        group_members = torch.cat(group_members, dim=0)
                        centroid = torch.mean(group_members, dim=0, keepdim=True)
                        centroids.append(centroid)

                if len(centroids) > 0:
                    centroids = torch.cat(centroids, dim=0)
                    centroids = torch.mean(centroids, dim=0)
                else:
                    centroids = torch.mean(merged_features[b], dim=0)

                batch_centroids.append(centroids)

            except Exception as e:
                print(f"Warning: Grouping failed for batch {b}: {str(e)}, using mean as fallback")
                centroids = torch.mean(merged_features[b], dim=0)
                batch_centroids.append(centroids)

        batch_centroids = torch.stack(batch_centroids, dim=0)

        batch_centroids = F.normalize(batch_centroids, dim=1, eps=1e-8)

        return batch_centroids

class IOZ_RAD(nn.Module):
    def __init__(self, pretrained_backbone: CLIP, lang_dim: int, vis_dim: int,
                 ctx_length: int, ctx_depth: int, ctx_branch: str, ctx_type: str,
                 use_msfa: bool, n_groups: int,
                 tap_layers: list, device: str, spatial_size: int,
                 parallel_mode=True, n_parallel=4, act_fn='gelu', normalize=True, drop_ratio=0.1,
                 lang_adapt_layers=6, lang_adapt_alpha=0.1, vis_cond_map_dim: int = 128,
                 enable_domain_align: bool = True, grl_strength: float = 0.1):
        super(IOZ_RAD, self).__init__()
        self.pretrained_backbone = pretrained_backbone

        self.visual = self.pretrained_backbone.visual
        self.transformer = self.pretrained_backbone.transformer
        self.token_embedding = self.pretrained_backbone.token_embedding
        self.positional_embedding = self.pretrained_backbone.positional_embedding
        self.ln_final = self.pretrained_backbone.ln_final
        self.text_projection = self.pretrained_backbone.text_projection
        self.attn_mask = self.pretrained_backbone.attn_mask

        self.tap_layers = tap_layers

        self.ctx_branch = ctx_branch
        self.ctx_type = ctx_type
        self.ctx_depth = ctx_depth
        self.ctx_length = ctx_length
        self.use_msfa = use_msfa
        self.n_groups = n_groups

        self.parallel_mode = parallel_mode
        self.n_parallel = n_parallel
        self.act_fn = act_fn
        self.normalize = normalize
        self.drop_ratio = drop_ratio

        if 'L' in self.ctx_branch:
            self.lang_ctx_on = True
        else:
            self.lang_ctx_on = False

        if 'V' in self.ctx_branch:
            self.vis_ctx_on = True
        else:
            self.vis_ctx_on = False

        self.linguistic_encoder = LinguisticEncoderModule(frozen=(not self.lang_ctx_on))
        self.lang_injector = ContextInjectionModule(lang_dim, ctx_length, ctx_depth, text_mode=True,
                                         ctx_type=ctx_type,
                                         active=self.lang_ctx_on)
        self.vis_injector = ContextInjectionModule(vis_dim, ctx_length, ctx_depth, text_mode=False,
                                           ctx_type=ctx_type,
                                           active=self.vis_ctx_on)

        self.spatial_mapper = FeatureMapper(
            vis_dim,
            lang_dim,
            len(tap_layers),
            do_stack=False,
            array_input=True,
            parallel_mode=parallel_mode,
            n_parallel=n_parallel,
            act_fn=act_fn,
            normalize=normalize,
            drop_ratio=drop_ratio
        )

        self.global_mapper = FeatureMapper(
            lang_dim,
            lang_dim,
            1,
            do_stack=False,
            array_input=False,
            parallel_mode=parallel_mode,
            n_parallel=n_parallel,
            act_fn=act_fn,
            normalize=normalize,
            drop_ratio=drop_ratio
        )

        if 'D' in self.ctx_type:
            self.adaptive_vis_ctx_gen = FeatureMapper(
                lang_dim,
                vis_dim,
                ctx_length,
                do_stack=True,
                array_input=False,
                parallel_mode=parallel_mode,
                n_parallel=n_parallel,
                act_fn=act_fn,
                normalize=normalize,
                drop_ratio=drop_ratio
            )

            self.adaptive_lang_ctx_gen = FeatureMapper(
                lang_dim,
                lang_dim,
                ctx_length,
                do_stack=True,
                array_input=False,
                parallel_mode=parallel_mode,
                n_parallel=n_parallel,
                act_fn=act_fn,
                normalize=normalize,
                drop_ratio=drop_ratio
            )

        if self.use_msfa:
            self.msfa = MultiScaleFeatureAggregation(n_groups)

        self.spatial_size = spatial_size
        self.device = device

        self.lang_adapt_layers = lang_adapt_layers
        self.lang_adapt_alpha = lang_adapt_alpha
        self.vis_cond_map_dim = vis_cond_map_dim

        self.enable_domain_align = enable_domain_align
        self.grl_strength = grl_strength
        self.cached_domain_losses = None

        if self.enable_domain_align:
            classifier_dim = lang_dim
            self.domain_classifier = DomainClassifierNet(feat_dim=classifier_dim, mid_dim=classifier_dim // 2)
            self.grl_module = GradFlipModule()
        else:
            self.domain_classifier = None
            self.grl_module = None

        if self.lang_adapt_layers > 0 and self.vis_cond_map_dim > 0:
            self.vis_cond_proj = nn.Linear(lang_dim, self.vis_cond_map_dim)
            refiner_in_dim = lang_dim + self.vis_cond_map_dim
        else:
            self.vis_cond_proj = None
            refiner_in_dim = lang_dim

        self.lang_refiners = nn.ModuleList([
            ProjectionBranch(
                refiner_in_dim,
                lang_dim,
                n_parallel=n_parallel,
                parallel_mode=parallel_mode,
                normalize=normalize,
                act_fn=act_fn,
                drop_ratio=drop_ratio
            ) for _ in range(self.lang_adapt_layers)
        ])

        self._setup_parameters()

    def _setup_parameters(self):
        for p in self.lang_refiners.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if hasattr(self, 'vis_cond_proj') and self.vis_cond_proj is not None:
            for p in self.vis_cond_proj.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)

        if hasattr(self, 'domain_classifier') and self.domain_classifier is not None:
            for m in self.domain_classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def prepare_adaptive_contexts(self, image):
        with torch.no_grad():
            vis_feats, _ = self.visual.forward(image, self.tap_layers)

        adaptive_vis_ctx = self.adaptive_vis_ctx_gen(vis_feats)
        adaptive_lang_ctx = self.adaptive_lang_ctx_gen(vis_feats)

        self.vis_injector.update_generated_ctx(adaptive_vis_ctx)
        self.lang_injector.update_generated_ctx(adaptive_lang_ctx)


    def extract_visual(self, image):

        x = image
        if self.visual.input_patchnorm:
            x = x.reshape(x.shape[0], x.shape[1],
                          self.visual.grid_size[0],
                          self.visual.patch_size[0],
                          self.visual.grid_size[1],
                          self.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.visual.grid_size[0] * self.visual.grid_size[1], -1)
            x = self.visual.patchnorm_pre_ln(x)
            x = self.visual.conv1(x)
        else:
            x = self.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)

        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        spatial_embed = x

        x = x.permute(1, 0, 2)

        spatial_tokens = []

        for idx, r in enumerate(self.visual.transformer.resblocks):
            x, tokens, attn_out = self.vis_injector(r, idx, x, k_x=None, v_x=None, attn_mask=None)

            if (idx + 1) in self.tap_layers:
                spatial_tokens.append(tokens)

        x = x.permute(1, 0, 2)
        spatial_tokens = [spatial_tokens[t].permute(1, 0, 2) for t in range(len(spatial_tokens))]

        if self.visual.attn_pool is not None:
            x = self.visual.attn_pool(x)
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        else:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj

        return pooled, spatial_tokens, spatial_embed

    def map_visual_tokens(self, vis_feats, spatial_tokens):

        mapped_patches = self.spatial_mapper(spatial_tokens)
        for layer in range(len(mapped_patches)):
            if torch.isnan(mapped_patches[layer]).any() or torch.isinf(mapped_patches[layer]).any():
                mapped_patches[layer] = torch.where(
                    torch.isnan(mapped_patches[layer]) | torch.isinf(mapped_patches[layer]),
                    torch.zeros_like(mapped_patches[layer]),
                    mapped_patches[layer]
                )

            mapped_patches[layer] = F.normalize(mapped_patches[layer], dim=-1, eps=1e-8)

        mapped_cls = self.global_mapper(vis_feats)[0]

        if torch.isnan(mapped_cls).any() or torch.isinf(mapped_cls).any():
            mapped_cls = torch.where(
                torch.isnan(mapped_cls) | torch.isinf(mapped_cls),
                torch.zeros_like(mapped_cls),
                mapped_cls
            )

        mapped_cls = F.normalize(mapped_cls, dim=-1, eps=1e-8)

        return mapped_cls, mapped_patches

    def extract_linguistic(self, text, vis_conditioning: Optional[torch.Tensor] = None):
        compute_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(compute_dtype)

        x = x + self.positional_embedding.to(compute_dtype)
        x = x.permute(1, 0, 2)

        for idx, r in enumerate(self.transformer.resblocks):
            x, attn_out = self.lang_injector(r, idx, x, k_x=None, v_x=None, attn_mask=self.attn_mask)

            if idx < self.lang_adapt_layers:
                residual_ref = x
                adapter_input = residual_ref

                if self.vis_cond_proj is not None and vis_conditioning is not None:
                    mapped_vis_cond = self.vis_cond_proj(vis_conditioning)

                    L_seq, N_batch, _ = residual_ref.shape

                    if N_batch != mapped_vis_cond.shape[0]:
                         if mapped_vis_cond.shape[0] == 1:
                             mapped_vis_cond = mapped_vis_cond.expand(N_batch, -1)
                         else:
                             pass

                    broadcast_vis_cond = mapped_vis_cond.unsqueeze(0).expand(L_seq, N_batch, -1)
                    adapter_input = torch.cat([residual_ref, broadcast_vis_cond], dim=-1)

                refined_out = self.lang_refiners[idx](adapter_input)

                scaled_refined = (
                    refined_out
                    * residual_ref.norm(dim=-1, keepdim=True)
                    / (refined_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = self.lang_adapt_alpha * scaled_refined + (1 - self.lang_adapt_alpha) * residual_ref

        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def cross_modal_scoring(self, vis_feat, spatial_tok, lang_feat, do_merge):
        deviation_maps = []

        for layer in range(len(spatial_tok)):
            if torch.isnan(spatial_tok[layer]).any() or torch.isinf(spatial_tok[layer]).any():
                print(f"Warning: spatial_tok[{layer}] contains NaN/Inf, cleaning...")
                spatial_tok[layer] = torch.where(
                    torch.isnan(spatial_tok[layer]) | torch.isinf(spatial_tok[layer]),
                    torch.zeros_like(spatial_tok[layer]),
                    spatial_tok[layer]
                )

            if torch.isnan(lang_feat).any() or torch.isinf(lang_feat).any():
                print("Warning: lang_feat contains NaN/Inf, cleaning...")
                lang_feat = torch.where(
                    torch.isnan(lang_feat) | torch.isinf(lang_feat),
                    torch.zeros_like(lang_feat),
                    lang_feat
                )

            deviation_map = (100.0 * spatial_tok[layer] @ lang_feat)

            if torch.isnan(deviation_map).any() or torch.isinf(deviation_map).any():
                print(f"Warning: deviation_map[{layer}] contains NaN/Inf after multiplication")
                deviation_map = torch.where(
                    torch.isnan(deviation_map) | torch.isinf(deviation_map),
                    torch.zeros_like(deviation_map),
                    deviation_map
                )

            deviation_maps.append(deviation_map)

        if self.use_msfa:
            try:
                avg_vals = [dm.mean() for dm in deviation_maps]
                avg_vals = [v if torch.isfinite(v) else torch.tensor(0.0, device=v.device) for v in avg_vals]
                blend_ratio = torch.sigmoid(torch.mean(torch.stack(avg_vals)))
            except Exception as e:
                print(f"Warning: Failed to compute blend_ratio: {e}, using default 0.5")
                blend_ratio = torch.tensor(0.5, device=vis_feat.device)

            try:
                aggregated_feat = self.msfa.forward(spatial_tok, deviation_maps)

                enhanced_vis = blend_ratio * aggregated_feat + (1 - blend_ratio) * vis_feat

                cosine_sim = F.cosine_similarity(enhanced_vis, vis_feat, dim=1, eps=1e-8).unsqueeze(1)
                blend_gate = torch.sigmoid(cosine_sim * 5.0)
                enhanced_vis = blend_gate * enhanced_vis + (1 - blend_gate) * vis_feat

                enhanced_vis = F.normalize(enhanced_vis, dim=1, eps=1e-8)
            except Exception as e:
                print(f"Warning: MSFA processing failed: {e}, using original features")
                enhanced_vis = vis_feat
        else:
            enhanced_vis = vis_feat

        deviation_score = (100.0 * enhanced_vis.unsqueeze(1) @ lang_feat)
        deviation_score = deviation_score.squeeze(1)

        if torch.isnan(deviation_score).any() or torch.isinf(deviation_score).any():
            print("Warning: deviation_score contains NaN/Inf before softmax")
            deviation_score = torch.where(
                torch.isnan(deviation_score) | torch.isinf(deviation_score),
                torch.zeros_like(deviation_score),
                deviation_score
            )

        deviation_score = torch.softmax(deviation_score, dim=1)

        upsampled_maps = []
        for i in range(len(deviation_maps)):
            B, L, C = deviation_maps[i].shape
            H = int(np.sqrt(L))

            resized_map = deviation_maps[i].permute(0, 2, 1).view(B, 2, H, H)

            try:
                from torch.nn.functional import gaussian_blur
                resized_map = gaussian_blur(resized_map, kernel_size=[3, 3], sigma=[1.0, 1.0])
            except ImportError:
                pass

            resized_map = F.interpolate(resized_map, size=self.spatial_size, mode='bilinear', align_corners=True)

            upsampled_maps.append(resized_map)

        if do_merge:
            layer_weights = F.softmax(torch.tensor([i+1 for i in range(len(upsampled_maps))],
                                            device=upsampled_maps[0].device, dtype=torch.float), dim=0)

            deviation_map = torch.zeros_like(upsampled_maps[0])
            for i, dm in enumerate(upsampled_maps):
                deviation_map += layer_weights[i] * dm

            deviation_map = torch.softmax(deviation_map, dim=1)

            deviation_map = (deviation_map[:, 1:, :, :] + 1 - deviation_map[:, 0:1, :, :]) / 2.0
            deviation_score = deviation_score[:, 1]

            return deviation_map, deviation_score
        else:
            for i in range(len(upsampled_maps)):
                upsampled_maps[i] = torch.softmax(upsampled_maps[i], dim=1)

            return upsampled_maps, deviation_score

    def compute_representations(self, image, category_label):
        if 'D' in self.ctx_type:
            self.prepare_adaptive_contexts(image)

        pooled_feats, spatial_tokens, _ = self.extract_visual(image)

        mapped_cls, mapped_patches = self.map_visual_tokens(pooled_feats, spatial_tokens)

        if self.lang_ctx_on:
            lang_feats = self.linguistic_encoder(self, category_label, self.device, vis_conditioning_batch=mapped_cls)
        else:
            with torch.no_grad():
                lang_feats = self.linguistic_encoder(self, category_label, self.device, vis_conditioning_batch=mapped_cls)


        return mapped_cls, mapped_patches, lang_feats, pooled_feats

    @torch.cuda.amp.autocast()
    def forward(self, image, category_label, do_merge=True):
        vis_feats, spatial_tokens, lang_feats, pooled_vis_baseline = self.compute_representations(image, category_label)

        if self.enable_domain_align and self.domain_classifier is not None:
            shifted_feats = vis_feats
            baseline_feats = pooled_vis_baseline

            disc_out_baseline = self.domain_classifier(baseline_feats.detach())
            loss_disc_base = F.binary_cross_entropy_with_logits(
                disc_out_baseline, torch.zeros_like(disc_out_baseline, device=disc_out_baseline.device)
            )

            disc_out_shifted = self.domain_classifier(shifted_feats.detach())
            loss_disc_shift = F.binary_cross_entropy_with_logits(
                disc_out_shifted, torch.ones_like(disc_out_shifted, device=disc_out_shifted.device)
            )
            loss_domain_disc = (loss_disc_base + loss_disc_shift) * 0.5

            grl_feats = self.grl_module.apply(shifted_feats, self.grl_strength)
            disc_out_grl = self.domain_classifier(grl_feats)
            loss_domain_adv = F.binary_cross_entropy_with_logits(
                disc_out_grl, torch.zeros_like(disc_out_grl, device=disc_out_grl.device)
            )

            self.cached_domain_losses = (loss_domain_disc, loss_domain_adv)
        else:
            self.cached_domain_losses = None

        deviation_map, deviation_score = self.cross_modal_scoring(vis_feats, spatial_tokens, lang_feats, do_merge)

        if do_merge:
            deviation_map = deviation_map
            deviation_score = deviation_score
            deviation_map = deviation_map.squeeze(1)

            return deviation_map, deviation_score
        else:
            deviation_maps = deviation_map
            deviation_score = deviation_score

            return deviation_maps, deviation_score
