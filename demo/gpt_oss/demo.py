"""GPT-OSS-20B on MPK. Pass --use-mirage for the megakernel, omit it for the
HuggingFace reference.

The megakernel runs the released checkpoint: its MXFP4 expert weights stream
as they are and are converted to bf16 exactly inside the expert GEMMs (--moe
plan, the default). A bf16 checkpoint (Mxfp4Config(dequantize=True)) runs too,
and is what the HuggingFace reference and --moe legacy need.
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import mirage as mi
from mirage.mpk.kv_planner import resolve_pool_size
from mirage.mpk.models.gpt_oss.builder import GptOssBuilder, plan_kv_cache

DEFAULT_PROMPT = "Give me a short introduction to large language models."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true",
                        help="Use Mirage kernels")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                        help="The released checkpoint (MXFP4 experts, run as "
                             "they are) or a local bf16 one")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--raw-prompt", action="store_true",
                        help="Skip the chat template")
    parser.add_argument("--max-seq-length", default=512, type=int,
                        help="Max sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Decode cap; defaults to filling max_seq_length")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int,
                        help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int,
                        help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=64, type=int,
                        help="Tokens per page for the KV stream with largest per-token "
                             "footprint (same for the two KV streams in GPT-OSS).")
    parser.add_argument("--kv-budget", type=str, default=None,
                        help="Memory budget for KV cache as a size('24GiB')."
                             "Exclusive with --max-num-pages")
    parser.add_argument("--max-num-pages", default=None, type=int,
                        help="Max num pages. Exclusive with --kv-budget")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Ignore eos token during generation")
    parser.add_argument("--moe", choices=["plan", "legacy"], default="plan",
                        help="plan: route plan + packed expert GEMMs (the "
                             "released MXFP4 checkpoint or a bf16 one); "
                             "legacy: the bf16-only MoE linear layers")
    parser.add_argument("--dense", choices=["mpk", "gemm"], default="mpk",
                        help="the other linear layers: MPK's (mpk) or the "
                             "expert GEMM task's dense mode (gemm, with "
                             "--moe plan; kept for comparison)")
    args = parser.parse_args()

    print("Input arguments:", args)
    torch.set_default_dtype(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.raw_prompt:
        text = args.prompt
    else:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids[0]
    # Benchmark hook: MPK_BENCH_PROMPT_IDS names a raw int32 file of
    # [requests, prompt_len] token ids. The megakernel then serves
    # --max-num-batched-requests requests, each with its own raw prompt,
    # instead of the single prompt above.
    bench_prompt_ids = os.environ.get("MPK_BENCH_PROMPT_IDS")
    bench_ids = None
    if bench_prompt_ids:
        import numpy as np
        bench_ids = torch.from_numpy(
            np.fromfile(bench_prompt_ids, dtype=np.int32)
            .reshape(args.max_num_batched_requests, -1).astype(np.int64))
        input_ids = bench_ids[0]
    prompt_len = input_ids.shape[-1]
    assert prompt_len < args.max_seq_length
    output_len = (args.max_new_tokens if args.max_new_tokens is not None
                  else args.max_seq_length - prompt_len)

    starter, ender = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))

    if args.use_mirage:
        # The megakernel stops at max_seq_length, so shorten it to make
        # --max-new-tokens the binding limit.
        seq_len = min(args.max_seq_length, prompt_len + output_len)
        if bench_ids is not None:
            tokens = torch.zeros(bench_ids.shape[0], seq_len, dtype=torch.long,
                                 device="cuda")
            tokens[:, :prompt_len] = bench_ids.to("cuda")
        else:
            tokens = torch.zeros(1, seq_len, dtype=torch.long, device="cuda")
            tokens[0, :prompt_len] = input_ids.to("cuda")
        mbt = args.max_num_batched_tokens
        n_req = tokens.shape[0]
        # KV 2.0: the plan is the source of truth for the cache layout
        config = AutoConfig.from_pretrained(args.model)
        kv_plan = plan_kv_cache(config, args.page_size)
        try:
            max_num_pages = resolve_pool_size(
                kv_plan, kv_budget=args.kv_budget,
                max_num_pages=args.max_num_pages, max_seq_length=seq_len,
                max_num_batched_requests=args.max_num_batched_requests,
                max_num_batched_tokens=mbt)
        except ValueError as e:
            raise SystemExit(str(e))
        meta_tensors = {
            "step": torch.zeros(n_req, dtype=torch.int32, device="cuda"),
            "tokens": tokens,
            "input_tokens": torch.zeros(mbt, 1, dtype=torch.long, device="cuda"),
            "output_tokens": torch.zeros(mbt, 1, dtype=torch.long, device="cuda"),
            "num_new_tokens": torch.ones(n_req, dtype=torch.int32, device="cuda"),
            "prompt_lengths": torch.full((n_req,), prompt_len,
                                         dtype=torch.int32, device="cuda"),
            "qo_indptr_buffer": torch.empty(args.max_num_batched_requests + 1,
                                            dtype=torch.int32, device="cuda"),
            **kv_plan.build_meta_tensors(
                max_num_batched_requests=args.max_num_batched_requests,
                max_seq_length=seq_len),
        }

        num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
        # The task graph is queued once an iteration, so the schedulers have
        # little to do: on a 148-SM GPU 144 workers and 16 schedulers (4
        # CTAs) instead of 128 and 64. Worth little: GPT-OSS at batch
        # 1 / 4 / 16 363 / 530 / 791 -> 375 / 515 / 739 ms, Qwen3.5 the same
        # either way (with 4 schedulers the queueing slows batch 1 down).
        if torch.cuda.get_device_properties(0).multi_processor_count >= 148:
            num_workers, num_schedulers = 144, 16
        # Benchmark hook: MPK_BENCH_WORKERS="<workers>,<schedulers>" overrides
        # the split of the GPU's SMs (schedulers in blocks of 4 a CTA).
        if os.environ.get("MPK_BENCH_WORKERS"):
            num_workers, num_schedulers = (
                int(v) for v in os.environ["MPK_BENCH_WORKERS"].split(","))
        # Benchmark hook: MPK_BENCH_PROFILE names a .npy file that receives the
        # raw profiler buffer (the built-in exporters are too slow for a whole
        # generation).
        bench_profile = os.environ.get("MPK_BENCH_PROFILE")
        profiler_tensor = None
        if bench_profile:
            profiler_tensor = torch.zeros(
                int(os.environ.get("MPK_BENCH_PROFILE_ENTRIES", str(1 << 24))),
                dtype=torch.uint64, device="cuda")
        mpk = mi.PersistentKernel(
            mode="offline", world_size=1, mpi_rank=0,
            num_workers=num_workers, num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=seq_len,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=mbt,
            max_num_pages=max_num_pages,
            kv_groups=kv_plan.group_specs(),
            eos_token_id=-1 if args.ignore_eos else 200002,
            meta_tensors=meta_tensors, profiler_tensor=profiler_tensor, trace_name="",
            spec_decode_config=None, use_cutlass_kernel=False,
        )

        print("Building the task graph...")
        GptOssBuilder(mpk, kv_plan=kv_plan, moe=args.moe,
                      dense=args.dense).build_from_model(
            model_name=args.model, model_path=args.model)
        print("Compiling the megakernel...")
        mpk.compile(output_dir=args.output_dir)
        if bench_profile:
            mpk.profiler_tensor = None

        # Benchmark hook: MPK_BENCH_WARMUP_MS of unrelated GPU work right before
        # the launch, so the launch does not start on an idle GPU's clocks.
        bench_warm_ms = float(os.environ.get("MPK_BENCH_WARMUP_MS", "0"))
        if bench_warm_ms > 0:
            bench_warm_end = time.perf_counter() + bench_warm_ms / 1e3
            bench_warm = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
            while time.perf_counter() < bench_warm_end:
                (bench_warm @ bench_warm).sum().item()
            del bench_warm
        # Benchmark hook: MPK_BENCH_WATCH=<seconds> polls the requests'
        # positions during the launch and gives up, with a report of what the
        # workers last did, when they stop moving for that long.
        bench_watch = float(os.environ.get("MPK_BENCH_WATCH", "0"))
        if bench_watch > 0:
            from mirage.mpk.bench_watch import LaunchWatch
            watch = LaunchWatch(meta_tensors["step"], profiler_tensor,
                                num_workers)
        torch.cuda.synchronize()
        bench_t0, bench_c0 = time.time(), time.perf_counter()
        starter.record()
        if bench_watch > 0:
            mpk.launch_func(int(torch.cuda.current_stream().cuda_stream))
            ender.record()
            if not watch.wait(ender, bench_watch):
                os._exit(3)
        else:
            mpk()
            ender.record()
        torch.cuda.synchronize()
        bench_wall_ms = (time.perf_counter() - bench_c0) * 1e3
        run_time = starter.elapsed_time(ender)
        if bench_profile:
            import numpy as np
            np.save(bench_profile, profiler_tensor.cpu().numpy())
        if bench_prompt_ids:
            # Benchmark hook: host wall clock of the one launch that holds the
            # whole generation, and optionally the tokens it wrote.
            if os.environ.get("MPK_BENCH_TOKENS_OUT"):
                tokens.cpu().numpy().astype("int32").tofile(os.environ["MPK_BENCH_TOKENS_OUT"])
            print("MPK_BENCH " + json.dumps({
                "wall_ms": bench_wall_ms, "event_ms": run_time,
                "t0": bench_t0, "t1": bench_t0 + bench_wall_ms / 1e3,
                "requests": n_req, "prompt_len": int(prompt_len),
                "seq_len": int(seq_len),
                "step_min": int(meta_tensors["step"].min().item()),
                "step_max": int(meta_tensors["step"].max().item()),
            }))

        end_idx = min(meta_tensors["step"][0].item() + 1,
                      prompt_len + output_len)
        token_ids = tokens[0, prompt_len:end_idx].cpu().tolist()
        response = tokenizer.decode(tokens[0, :end_idx],
                                    skip_special_tokens=True)
        mpk.finalize()
    else:
        print("Loading the HuggingFace reference...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda").eval()

        starter.record()
        inp = input_ids.unsqueeze(0).cuda()
        with torch.no_grad():
            out = model.generate(
                inp,
                # pad_token_id == eos_token_id here, so generate() cannot infer
                # the mask; pass it explicitly to avoid the fallback
                attention_mask=torch.ones_like(inp),
                max_new_tokens=output_len, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                eos_token_id=(None if args.ignore_eos
                              else model.config.eos_token_id),
                pad_token_id=tokenizer.pad_token_id or 199999)
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        token_ids = out[0, prompt_len:].cpu().tolist()
        response = tokenizer.decode(out[0], skip_special_tokens=True)

    tokens_generated = len(token_ids)
    per_tok_ms = run_time / max(prompt_len + tokens_generated, 1)
    print(response)
    print("Prompt length {}, generate length {}, per-token latency {:.3f} ms"
          .format(prompt_len, tokens_generated, per_tok_ms))
