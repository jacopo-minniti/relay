# Fast-dLLM v2 with RELAY (paper: *Learned Relay Representations*)

This directory is a fork of the upstream
[Fast-dLLM v2](https://github.com/NVlabs/Fast-dLLM) `v2/` codebase (Wu et al.,
[arXiv:2509.26328](https://arxiv.org/abs/2509.26328)) that adds the **RELAY**
extension introduced in *Learned Relay Representations for Forward-Thinking
Discrete Diffusion Models*. Use it to reproduce the **Vanilla SFT**,
**Rollout**, **RELAY (sg)**, and **RELAY** rows of Table 2 (paper,
Section 4.2) on HumanEval / HumanEval+ and MBPP / MBPP+.

> **Canonical setup, training, and evaluation commands** live in the
> top-level [relay README](../../README.md). This file documents only what is
> specific to the Fast-dLLM v2 side.

## What changed vs. upstream Fast-dLLM v2

The fork is intentionally minimal: we only touch what RELAY needs.

- **`src/lmflow/models/fast_dllm/`**
  - `configuration.py`: adds `use_relay` (bool, default `False`) and
    `relay_layer` (int, default `-1` = last decoder layer) to
    `Fast_dLLM_QwenConfig`.
  - `modeling.py`: adds a single zero-init `nn.LayerNorm` named
    `relay_layer_norm` and an additive injection at mask-token positions
    only (`x = token_emb + relay_layer_norm(h_t)`, paper Algorithm 1
    line 7); reads off `relay_layer`'s decoder-layer hidden state as the
    next-step relay state `h_s`.
- **`src/lmflow/pipeline/utils/`**
  - `block_bptt_loss.py` (`FastDLLMBlockBPTTLoss`): the 2-step rollout
    loss (paper Algorithm 1) wired into Fast-dLLM v2's BD3-LM-style
    doubled-attention training.
  - `streaming_batch.py` (`StreamingBatch`): the per-rank rollout buffer
    that persists `(x_input, h_s)` across calls so the loss runs on the
    full unmasking trajectory, not only the high-mask tail.
  - `bptt_trainer.py` (`FastDLLMBPTTTrainer`): a `transformers.Trainer`
    subclass that swaps the in-model MDM loss for the rollout loss and
    logs the relay-only gradient-norm ratio.
- **`src/lmflow/pipeline/finetuner.py`**: gates the new trainer behind
  `--loss_type bptt` and attaches the relay LayerNorm at run start when
  loading a checkpoint that does not yet have one.
- **`src/lmflow/args.py`**: the `bptt_*` flags
  (`bptt_use_relay`, `bptt_stop_grad_h_s`, `bptt_relay_layer`,
  `bptt_threshold`, `bptt_top_p`, `bptt_temperature`,
  `bptt_unmask_strategy`, `bptt_inner_block_size`).
- **`scripts/`** and **`train_scripts/`**: trimmed to the OpenCodeInstruct
  + OpenMathInstruct-2 c40m60 mixture used for Table 2 and the EvalPlus
  (HumanEval / MBPP) eval pipeline. The launch helpers
  (`scripts/launch_opencode_openmath_c40m60_4run.sh`) submit the three
  reported configurations (`vanilla`, `relay_sg`, `relay`) concurrently.
- **`eval.py`** and **`generation_functions.py`**: gain a `--use_carry`
  switch that turns on the 2-step relay-state carry at inference.

The upstream Fast-dLLM v2 inference pipeline (block-diffusion, KV cache,
parallel sub-block decoding, threshold-based unmasking) is untouched.

## Reproducibility checklist

- **Base checkpoint**:
  [`Efficient-Large-Model/Fast_dLLM_v2_1.5B`](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B)
- **Released RELAY checkpoints** (post-trained from the base, 200 optimizer
  steps on the c40m60 mixture):
  - **RELAY**:
    [`brozonoyer/relay-fastdllm-v2-c40m60-relay-step200`](https://huggingface.co/brozonoyer/relay-fastdllm-v2-c40m60-relay-step200)
  - **RELAY (sg)**:
    [`brozonoyer/relay-fastdllm-v2-c40m60-relay-sg-step200`](https://huggingface.co/brozonoyer/relay-fastdllm-v2-c40m60-relay-sg-step200)

  Both repos bundle `auto_map`-wired `configuration.py` / `modeling.py`, so
  `AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)`
  loads them directly with canonical RELAY config and weight keys—same as
  fresh `checkpoint-*` dirs produced by training in this tree.
- **Eval framework**: [EvalPlus](https://github.com/evalplus/evalplus) at
  the version pinned in [`requirements.txt`](requirements.txt)
  (`evalplus==<pinned>`). Threshold `0.85`, BD block 32, sub-block 8,
  reported on HumanEval / HumanEval+ / MBPP / MBPP+ exactly as in
  Wu et al. (2025b).
- **W&B**: the eval runs whose Base/Plus/NFE numbers populate Table 2 are
  hosted on a project under the authors' W&B entity; the entity name is
  withheld during the double-blind review period.

## Citation

The RELAY adaptation is described in our submission; please also cite the
underlying Fast-dLLM v2 work:

```bibtex
@misc{wu2025fastdllmv2efficientblockdiffusion,
      title={Fast-dLLM v2: Efficient Block-Diffusion LLM},
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Shizhe Diao and Yonggan Fu and Zhijian Liu and Pavlo Molchanov and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2509.26328},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.26328},
}
```
