import json
import os

import torch

from ..graph_builder import GraphBuilder
from ..utils import grid_for_rmsnorm_linear_layer
from ..gpt_oss.builder import load_checkpoint_tensors
from ...kv_planner import KVCachePlan, KVSpec, plan_kv_groups
from ...persistent_kernel import PersistentKernel
from ...model_registry import register_model_builder
from ... import moe_pack
from ....core import bfloat16, float32, int32, int64

from typing import Optional


def load_text_config(source: str) -> dict:
    """The text model's config of a Qwen3.5 checkpoint (a directory or a hub
    id), read as JSON: the megakernel needs no transformers support for it."""
    if os.path.isdir(source):
        path = os.path.join(source, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(source, "config.json")
    cfg = json.load(open(path))
    return cfg.get("text_config", cfg)


def plan_kv_cache(config: dict, page_size: int) -> KVCachePlan:
    """One KV stream: the full-attention layers (every fourth layer). The
    gated delta net layers keep their own states."""
    per_entry_bytes = 2 * config["num_key_value_heads"] * config["head_dim"] * 2
    full = tuple(i for i, t in enumerate(config["layer_types"])
                 if t == "full_attention")
    return plan_kv_groups([
        KVSpec("full_attention", per_entry_bytes=per_entry_bytes,
               layer_ids=full, preferred_block_size=page_size)])


def _round_up(n: int, m: int) -> int:
    return (n + m - 1) // m * m


@register_model_builder("qwen3_5", "Qwen3.5", "Qwen/Qwen3.5-35B-A3B")
class Qwen35Builder(GraphBuilder):
    """Qwen3.5 (text): gated delta net layers with a full-attention layer
    every fourth, and a routed MoE with a shared expert in every layer.

    Decode runs one token a request (max_tokens_per_request = 1): a gated
    delta net layer's states advance one position a step.
    """

    def __init__(self, mpk: PersistentKernel, weights: Optional[dict] = None,
                 kv_plan: Optional[KVCachePlan] = None,
                 num_requests: Optional[int] = None, dense: str = "mpk"):
        """dense: the linear layers other than the experts' (the mixers'
        projections, the router, lm_head) -- "mpk": MPK's linear layers;
        "gemm": the expert GEMM task's dense mode, kept for comparison."""
        super().__init__(mpk, weights)
        assert dense in ("mpk", "gemm")
        self.dense_gemm = dense == "gemm"
        assert mpk.world_size == 1, "Qwen3.5 is single-GPU for now"
        assert mpk.max_tokens_per_request == 1, (
            "Qwen3.5 needs one token a request and step: set "
            "MPK_MAX_TOKENS_PER_REQUEST=1 (max_tokens_per_request=1)")
        self.max_num_pages = mpk.max_num_pages
        self.kv_plan = kv_plan
        # the gated delta net states are indexed by request id
        self.num_requests = num_requests or mpk.max_num_batched_requests
        self.input_tokens = mpk.meta_tensors["input_tokens"]
        self.output_tokens = mpk.meta_tensors["output_tokens"]
        self.tokenizer = None
        self._keep = []

    # ---------------------------------------------------------------- loading

    def build_from_model(self, model_name: str, model_path: str | None = None):
        from transformers import AutoTokenizer

        source = model_path or model_name
        cfg = self.config = load_text_config(source)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.eos_token_id = cfg["eos_token_id"]
        self.hidden_size = cfg["hidden_size"]
        self.vocab_size = cfg["vocab_size"]
        self.num_layers = cfg["num_hidden_layers"]
        self.layer_types = list(cfg["layer_types"])
        self.eps = cfg["rms_norm_eps"]
        # full attention
        self.num_q_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.q_per_kv = self.num_q_heads // self.num_kv_heads
        self.head_dim = cfg["head_dim"]
        rope = cfg["rope_parameters"]
        self.rotary_dim = int(self.head_dim * rope.get("partial_rotary_factor", 1.0))
        # gated delta net
        self.gdn_key_heads = cfg["linear_num_key_heads"]
        self.gdn_dim = cfg["linear_key_head_dim"]
        assert cfg["linear_num_value_heads"] == 2 * self.gdn_key_heads
        assert cfg["linear_value_head_dim"] == self.gdn_dim
        assert cfg["linear_conv_kernel_dim"] == 4
        # MoE
        self.num_experts = cfg["num_experts"]
        self.topk = cfg["num_experts_per_tok"]
        self.moe_inter = cfg["moe_intermediate_size"]
        assert cfg["shared_expert_intermediate_size"] == self.moe_inter

        # RoPE on the first rotary_dim dims of a head. Text positions are the
        # same on the three mrope axes, so the tables are the plain ones.
        inv_freq = 1.0 / (rope["rope_theta"] ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) /
            self.rotary_dim))
        freqs = torch.arange(self.mpk.max_seq_length, dtype=torch.float)[:, None] * inv_freq
        self.cos_table = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(torch.bfloat16)
        self.sin_table = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(torch.bfloat16)

        assert self.kv_plan is not None, (
            "pass kv_plan=plan_kv_cache(config, page_size)")
        entry_shape = (self.num_kv_heads, self.head_dim)
        self.kv_pool, self.kv_views = self.kv_plan.allocate_pool(
            {g.spec_name: [("k", entry_shape, torch.bfloat16),
                           ("v", entry_shape, torch.bfloat16)]
             for g in self.kv_plan.groups},
            max_num_pages=self.max_num_pages)

        sd = load_checkpoint_tensors(source)
        prefix = "model.language_model."
        sd = {k[len(prefix):] if k.startswith(prefix) else k: v
              for k, v in sd.items()
              if k.startswith(prefix) or k == "lm_head.weight"}
        self.build_from_dict(sd)

    def build_from_config(self, model_config):
        raise NotImplementedError(
            "Qwen3.5 is built from a checkpoint; use build_from_model")

    # ------------------------------------------------------- weight conversion

    def _attach(self, tensor: torch.Tensor, name: str):
        tensor = tensor.contiguous().to("cuda")
        self._keep.append(tensor)
        return self.mpk.attach_input(torch_tensor=tensor, name=name)

    def _dense(self, weight: torch.Tensor, name: str, rows: int = None):
        """A linear layer's weight [rows, K]: as it is for MPK's linear layer
        (zero rows up to `rows`, a multiple of its 128-row tasks), or as
        dense_gemm_layer's tiles."""
        if self.dense_gemm:
            return self._attach(
                moe_pack.pack_bf16_tiles(weight.to("cuda")[None]), name)
        if rows is not None and weight.shape[0] != rows:
            weight = torch.cat(
                [weight, weight.new_zeros(rows - weight.shape[0],
                                          weight.shape[1])], dim=0)
        return self._attach(weight, name)

    def _gdn_weights(self, sd, p, i):
        """in_proj_qkv, in_proj_z, in_proj_b, in_proj_a as one projection
        (q | k | v | z | b | a rows, zero rows up to a multiple of 128)."""
        la = f"{p}linear_attn."
        w = torch.cat([sd[la + "in_proj_qkv.weight"], sd[la + "in_proj_z.weight"],
                       sd[la + "in_proj_b.weight"], sd[la + "in_proj_a.weight"]],
                      dim=0)
        par = torch.cat([-sd[la + "A_log"].float().exp(),
                         sd[la + "dt_bias"].float(),
                         sd[la + "norm.weight"].float()]).contiguous()
        channels = 4 * self.gdn_key_heads * self.gdn_dim
        nvh = 2 * self.gdn_key_heads
        conv_state = torch.zeros(self.num_requests, 4, channels,
                                 dtype=torch.bfloat16, device="cuda")
        state = torch.zeros(self.num_requests, nvh, self.gdn_dim, self.gdn_dim,
                            dtype=torch.float32, device="cuda")
        if getattr(self, "debug", None) is not None:
            self.debug[f"gdn_state_{i}"] = state
            self.debug[f"conv_state_{i}"] = conv_state
        return (self._dense(w, f"layer_{i}_gdn_proj", self.gdn_pw),
                self._attach(sd[la + "conv1d.weight"].reshape(channels, 4),
                             f"layer_{i}_gdn_conv"),
                self._attach(conv_state, f"layer_{i}_gdn_conv_state"),
                self._attach(state, f"layer_{i}_gdn_state"),
                self._attach(par, f"layer_{i}_gdn_params"),
                self._dense(sd[la + "out_proj.weight"], f"layer_{i}_gdn_out"))

    def _attn_weights(self, sd, p, i):
        """q_proj's query halves with k_proj and v_proj in the attention
        task's order (per KV head: its query heads, its key, its value), then
        q_proj's gate halves, as one projection."""
        sa = f"{p}self_attn."
        hd, h = self.head_dim, self.hidden_size
        q = sd[sa + "q_proj.weight"].reshape(self.num_q_heads, 2, hd, h)
        k = sd[sa + "k_proj.weight"].reshape(self.num_kv_heads, hd, h)
        v = sd[sa + "v_proj.weight"].reshape(self.num_kv_heads, hd, h)
        rows = []
        for g in range(self.num_kv_heads):
            rows.append(q[g * self.q_per_kv:(g + 1) * self.q_per_kv, 0]
                        .reshape(-1, h))
            rows.append(k[g])
            rows.append(v[g])
        rows.append(q[:, 1].reshape(-1, h))
        return (self._dense(torch.cat(rows, dim=0), f"layer_{i}_attn_proj",
                            self.attn_pw),
                self._attach(sd[sa + "q_norm.weight"], f"layer_{i}_q_norm"),
                self._attach(sd[sa + "k_norm.weight"], f"layer_{i}_k_norm"),
                self._dense(sd[sa + "o_proj.weight"], f"layer_{i}_o_proj"))

    def _moe_weights(self, sd, p, i):
        """The router with the shared expert's gate as one more row; the
        experts with the shared expert as one more expert, packed for
        moe_gemm_layer (gate and up rows interleaved)."""
        m = f"{p}mlp."
        router = torch.cat([sd[m + "gate.weight"],
                            sd[m + "shared_expert_gate.weight"]], dim=0)
        w13 = torch.cat(
            [sd[m + "experts.gate_up_proj"],
             torch.cat([sd[m + "shared_expert.gate_proj.weight"],
                        sd[m + "shared_expert.up_proj.weight"]], dim=0)[None]],
            dim=0).to("cuda")
        w2 = torch.cat([sd[m + "experts.down_proj"],
                        sd[m + "shared_expert.down_proj.weight"][None]],
                       dim=0).to("cuda")
        w13 = moe_pack.pack_bf16_tiles(moe_pack.interleave_gate_up(w13))
        w2 = moe_pack.pack_bf16_tiles(w2)
        small = self.dense_gemm and \
            self.mpk.max_num_batched_tokens * router.shape[0] <= 8192
        return ((self._attach(router, f"layer_{i}_router") if small
                 else self._dense(router, f"layer_{i}_router",
                                  self.router_rows)),
                self._attach(w13, f"layer_{i}_w13"),
                self._attach(w2, f"layer_{i}_w2"))

    # ------------------------------------------------------------------ graph

    def build_from_dict(self, sd: dict):
        mpk = self.mpk
        mbt = mpk.max_num_batched_tokens
        h, hd = self.hidden_size, self.head_dim
        nvh, gd = 2 * self.gdn_key_heads, self.gdn_dim
        slots = self.topk + 1
        e_tot = self.num_experts + 1
        self.qkv_size = self.num_kv_heads * (self.q_per_kv + 2) * hd
        self.attn_size = self.num_q_heads * hd
        # (MPK's linear layer: 128 output rows a task)
        pad = 1 if self.dense_gemm else 128
        self.attn_pw = _round_up(self.qkv_size + self.attn_size, pad)
        self.gdn_pw = _round_up(
            4 * self.gdn_key_heads * gd + nvh * gd + 2 * nvh, pad)
        self.router_rows = _round_up(e_tot, 8 if self.dense_gemm else 128)
        self.moe_nt = moe_pack.moe_tokens_per_pass(mbt)
        cap = moe_pack.moe_plan_cap(mbt, e_tot, slots, self.moe_nt)

        # Debugging aid: MPK_Q35_LAYERS builds only the first layers (then the
        # final norm and the head); MPK_Q35_DUMP keeps the residual stream's
        # buffers as torch tensors (self.debug, with the gated delta net
        # states) for the demo to save.
        num_layers = int(os.environ.get("MPK_Q35_LAYERS", self.num_layers))
        self.debug = {} if os.environ.get("MPK_Q35_DUMP") else None
        torch_dtype = {bfloat16: torch.bfloat16, float32: torch.float32,
                       int32: torch.int32, int64: torch.int64}

        def buf(name, dims, dtype=bfloat16):
            if self.debug is not None and name in (
                    "embed_out", "normed", "mixer_out", "layer_out",
                    "final_normed", "argmax_in", "moe_weight", "moe_logits",
                    "attn_proj", "attn_in", "attn_out", "attn_gated",
                    "gdn_proj", "gdn_out"):
                t = torch.zeros(dims, dtype=torch_dtype[dtype], device="cuda")
                self.debug[name] = t
                return mpk.attach_input(torch_tensor=t, name=name)
            return mpk.new_tensor(dims=dims, dtype=dtype, name=name,
                                  io_category="cuda_tensor")

        x = mpk.attach_input(torch_tensor=self.input_tokens, name="input_token")
        embed_out = buf("embed_out", (mbt, h))
        normed = buf("normed", (mbt, h))
        gdn_proj = buf("gdn_proj", (mbt, self.gdn_pw))
        gdn_out = buf("gdn_out", (mbt, nvh * gd))
        attn_proj = buf("attn_proj", (mbt, self.attn_pw))
        attn_in = buf("attn_in", (mbt, self.qkv_size))
        attn_out = buf("attn_out", (mbt, self.attn_size))
        attn_gated = buf("attn_gated", (mbt, self.attn_size))
        mixer_out = buf("mixer_out", (mbt, h))
        logits = buf("moe_logits", (mbt, self.router_rows))
        weight = buf("moe_weight", (mbt, slots), float32)
        route_list = buf("moe_route_list", (e_tot, mbt), int32)
        route_meta = buf("moe_route_meta", (1 + cap + e_tot,), int32)
        act = buf("moe_act", (mbt, slots, self.moe_inter))
        mlp_out = buf("moe_out", (mbt, slots, h))
        layer_out = buf("layer_out", (mbt, h))
        final_normed = buf("final_normed", (mbt, h))
        argmax_in = buf("argmax_in", (mbt, self.vocab_size))
        # (the argmax tasks split the vocabulary evenly: as many of them as
        # workers only if that divides it)
        argmax_tasks = max(g for g in range(1, mpk.num_workers + 1)
                           if self.vocab_size % g == 0)
        argmax_part_value = buf("argmax_part_value", (mbt, argmax_tasks))
        argmax_part_index = buf("argmax_part_index", (mbt, argmax_tasks), int64)
        argmax_out = mpk.attach_input(torch_tensor=self.output_tokens,
                                      name="output_token")
        cos_dt = self._attach(self.cos_table, "cos_position_embedding")
        sin_dt = self._attach(self.sin_table, "sin_position_embedding")
        norm_dummy = self._attach(torch.ones(hd, dtype=torch.bfloat16),
                                  "qk_norm_dummy")
        # dense="gemm": every GEMM is a moe_gemm task and the workers keep
        # their TMEM across them (tmem_release_layer at the end). Not with
        # MPK's linear layers, which allocate TMEM themselves.
        tmem = None
        if self.dense_gemm:
            tmem = self._attach(
                torch.zeros(mpk.num_workers, dtype=torch.int32), "tmem_token")
        # at small batches a wave has fewer units than workers: split every
        # unit's K steps (moe_gemm's KS) so that all of them work
        nt = self.moe_nt
        npass = (mbt + nt - 1) // nt
        active = min(e_tot, mbt * slots)
        rb = lambda rows: (rows + 127) // 128
        layers = {
            "w13": (active * rb(2 * self.moe_inter), mpk.gemm_steps(h, mbt),
                    cap * rb(2 * self.moe_inter)),
            "w2": (active * rb(h), mpk.gemm_steps(self.moe_inter, mbt),
                   cap * rb(h)),
        }
        if self.dense_gemm:
            layers.update({
                "gdn_proj": (npass * rb(self.gdn_pw), mpk.gemm_steps(h, mbt),
                             npass * rb(self.gdn_pw)),
                "attn_proj": (npass * rb(self.attn_pw), mpk.gemm_steps(h, mbt),
                              npass * rb(self.attn_pw)),
                "out": (npass * rb(h), mpk.gemm_steps(nvh * gd, mbt),
                        npass * rb(h)),
            })
        ks, scratch = mpk.gemm_split_plans(layers, nt)

        def linear(input, weight, output, rows, name, residual=None):
            """A dense linear layer: MPK's (128 output rows a task; 64 with a
            residual), or the expert GEMM task's dense mode."""
            if self.dense_gemm:
                mpk.dense_gemm_layer(
                    input=input, weight=weight, output=output, rows=rows,
                    residual=residual, tmem=tmem, ksplit=ks.get(name, 1),
                    scratch=scratch if ks.get(name, 1) > 1 else None)
            elif residual is not None:
                mpk.linear_with_residual_layer(
                    input=input, weight=weight, residual=residual,
                    output=output, grid_dim=(rows // 64, 1, 1),
                    block_dim=(256, 1, 1))
            else:
                mpk.linear_layer(
                    input=input, weight=weight, output=output,
                    grid_dim=(rows // 128, 1, 1), block_dim=(256, 1, 1))

        mpk.embed_layer(
            input=x, weight=self._attach(sd["embed_tokens.weight"], "embed_tokens"),
            output=embed_out, grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
            input_source=1)
        x = embed_out

        for i in range(num_layers):
            p = f"layers.{i}."
            mpk.norm1p_layer(
                x, self._attach(sd[p + "input_layernorm.weight"],
                                f"layer_{i}_input_norm"),
                normed, eps=self.eps)
            if self.layer_types[i] == "linear_attention":
                w_proj, conv_w, conv_state, state, par, w_out = \
                    self._gdn_weights(sd, p, i)
                linear(normed, w_proj, gdn_proj, self.gdn_pw, "gdn_proj")
                mpk.gdn_step_layer(gdn_proj, conv_w, conv_state, state, par,
                                   gdn_out, self.gdn_key_heads, gd, eps=self.eps)
                mixed, w_o = gdn_out, w_out
            else:
                w_proj, wq, wk, w_o = self._attn_weights(sd, p, i)
                linear(normed, w_proj, attn_proj, self.attn_pw, "attn_proj")
                mpk.qsa_prep_layer(attn_proj, wq, wk, attn_in,
                                   self.num_kv_heads, self.q_per_kv, hd,
                                   eps=self.eps)
                group_id, slot_id = self.kv_plan.layer_info(i)
                k_cache = mpk.attach_input(
                    torch_tensor=self.kv_plan.assert_in_pool(
                        self.kv_views[group_id]["k"][slot_id],
                        f"layer_{i} k_cache"),
                    name=f"layer_{i}_k_cache")
                v_cache = mpk.attach_input(
                    torch_tensor=self.kv_plan.assert_in_pool(
                        self.kv_views[group_id]["v"][slot_id],
                        f"layer_{i} v_cache"),
                    name=f"layer_{i}_v_cache")
                # (q and k arrive normalized: the norm's weight is 1 + w in
                # fp32, which the attention task's bf16 weights cannot hold)
                mpk.paged_attention_layer(
                    input=attn_in, k_cache=k_cache, v_cache=v_cache,
                    q_norm=norm_dummy, k_norm=norm_dummy,
                    cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
                    output=attn_out,
                    grid_dim=(mpk.max_num_batched_requests, self.num_kv_heads, 1),
                    block_dim=(256, 1, 1), enable_qk_norm=False,
                    rotary_dim=self.rotary_dim, group_id=group_id)
                mpk.gate_mul_layer(attn_out, attn_proj, self.qkv_size, attn_gated)
                mixed = attn_gated
            linear(mixed, w_o, mixer_out, h, "out", residual=x)
            x = mixer_out

            w_router, w13, w2 = self._moe_weights(sd, p, i)
            mpk.norm1p_layer(
                x, self._attach(sd[p + "post_attention_layernorm.weight"],
                                f"layer_{i}_post_norm"),
                normed, eps=self.eps)
            if self.dense_gemm and mbt * e_tot <= 8192:
                mpk.small_linear_layer(input=normed, weight=w_router,
                                       output=logits)
            elif self.dense_gemm:
                mpk.dense_gemm_layer(input=normed, weight=w_router,
                                     output=logits, rows=e_tot, tmem=tmem)
            else:
                linear(normed, w_router, logits, self.router_rows, "router")
            mpk.moe_route_layer(logits=logits, weight=weight,
                                route_list=route_list, route_meta=route_meta,
                                nt=self.moe_nt, num_experts=self.num_experts,
                                shared=True)
            mpk.moe_gemm_layer(
                input=normed, weight=w13, route_list=route_list,
                route_meta=route_meta, output=act, nt=self.moe_nt, mxfp4=False,
                rows=2 * self.moe_inter, act=(float("inf"), 1.0, 0.0),
                tmem=tmem, ksplit=ks["w13"], scratch=scratch)
            mpk.moe_gemm_layer(
                input=act, weight=w2, route_list=route_list,
                route_meta=route_meta, output=mlp_out, nt=self.moe_nt,
                mxfp4=False, rows=h, tmem=tmem, ksplit=ks["w2"],
                scratch=scratch)
            mpk.moe_combine_layer(input=mlp_out, weight=weight, residual=x,
                                  output=layer_out, shared=True)
            x = layer_out

        mpk.norm1p_layer(x, self._attach(sd["norm.weight"], "model_norm"),
                         final_normed, eps=self.eps)
        if self.dense_gemm:
            mpk.dense_gemm_layer(
                input=final_normed,
                weight=self._dense(sd["lm_head.weight"], "lm_head"),
                output=argmax_in, rows=self.vocab_size, tmem=tmem)
            mpk.tmem_release_layer(
                argmax_in, tmem,
                self._attach(torch.zeros(mpk.num_workers, dtype=torch.int32),
                             "tmem_done"))
        else:
            mpk.linear_layer(
                input=final_normed,
                weight=self._dense(sd["lm_head.weight"], "lm_head"),
                output=argmax_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(self.vocab_size), 1, 1),
                block_dim=(256, 1, 1))
        mpk.argmax_partial_layer(
            input=argmax_in, output=(argmax_part_value, argmax_part_index),
            grid_dim=(argmax_tasks, 1, 1), block_dim=(256, 1, 1))
        mpk.argmax_reduce_layer(
            input=(argmax_part_value, argmax_part_index), output=argmax_out,
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1))

    def encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, ids: torch.Tensor):
        return self.tokenizer.decode(ids, skip_special_tokens=True)
