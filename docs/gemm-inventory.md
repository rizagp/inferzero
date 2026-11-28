# GEMM inventory: the contract the shared GEMM layer must meet

Phase 0 of the move to one shared GEMM layer ([docs/design-shared-gemm.md](design-shared-gemm.md)), approved by the
user on 2026-11-27. The direction:
1. collect the fastest GEMM and GEMV forms, their scaling forms included;
2. the new route keeps every guard, bug fix and proven try / catch of the old paths;
3. Inkling 2 Mini is the first model on it, then its optimization;
4. then every model, the encoders, the tensor and the layer split, one at a time, the old path kept as a fallback
   until each is proven.

**The rule:** every row has its source and its proof, or says **"unproven"** and why. Nothing is left blank.

## A. Every GEMM / GEMV form in use (fused launches included: a fused form names what it fuses, see K)

Generated from the engine's own launch log (`ARC_GEMM_PLAN=1`): one line per distinct (weight role, weight type, row
class: one token / 2-8 rows / prompt chunk, host or resident) giving the kernel, its `-D` options, grid, local size and
caller. The source is one short run per model: a prompt of more than one chunk, 16 generated tokens, and the 2-8-row
passes. The laptop shows only Xe-LPG. Other GPU kinds' rows come from the contributor kit (J), and stay "unproven:
plan not seen on this kind" until then.

How to read the log:
- **Memory column:** a GEMV line says "raw" or "host raw", the rows it reads. A GEMM line says where its W8 comes from:
  "w8 resident", "w8 jit" or "host". "-" is a step with no weight.
- **One summary per model:** the "gemm-plan summary (freed)" line that runz prints at its end. C's harness runs runz, so
  that's the line it collects.
- **serverz:** prints "gemm-plan summary (after the warm-up)" after start-up. That holds only the routes the warm-up
  took, so it's partial. A stopped server prints no final summary: it ends without its destructors. A model switch
  prints the old model's "(freed)" summary.
- **A split model:** each shard logs its own lines and summary ("shard r"). serverz's warm-up summary is shard 0's or
  the last stage's.
- **Not in the log, listed by hand** until their migration turns (they bypass `Engine::run`):
  - Whisper's and Qwen3-ASR's own launchers (`zero/whisper_engine.h`, `zero/asr_qwen3.h`; kernels `whisper.cl` / `asr.cl`);
  - the NPU path (`zero/npu/engine_npu.h`: the prompt chunks it takes run on the NPU, not as GPU launches);
  - `engine_tp.h`'s one direct launch (the multi-GPU sums' setup).
- **Steps that aren't GEMMs** (norm, quant, swiglu, rot) give the model-file line ("from") too, since 43ab012.

### Xe-LPG (the laptop's 125H), generated

From `ARC_GEMM_PLAN=1` runs (the Developer, 2026-11-28; main 6fb85d7, driver 32.0.101.9175) of the five laptop models in the golden kinds that change routes: base, short, mtp, tp2, tp3 and layer (under `ARC_TP_SIM`), host. Each kind is skipped where the golden set skips it: Inkling runs base and short only, the 35B has no host. One row per kernel, options (numbers dropped; the type is its own column), row class, memory and fused form. Every plan line was parsed (per kind, lines with a kernel = plan lines). Kernels per model: Inkling-2-Mini 36, Qwen3.5-4B 30, Qwen3.5-9B 32, Qwen3.6-35B-A3B 42, Qwen3.8-27B 27. The exact options and local sizes, and each route's call site (`at` / `from`), are in the generated forms and routes files, kept with the run's logs, not in the repository.

Every row is the Xe-LPG kind only. On every other kind each form is "unproven: plan not seen on this kind" until the kit (J). The fastest measured form per row and its correctness proof are in K (fusions), C (hashes) and E (speed).

| Kernel | Options | Rows | Memory | Fused | Types | Roles | Taken by (model: kinds) |
|---|---|---|---|---|---|---|---|
| `asc_transpose` | - | 2-8 | - | - | - | moe_gateup | 35B mtp short; Ink short |
| `asc_transpose` | - | chunk | - | - | - | moe_gateup | 35B mtp tp2 tp3; Ink base |
| `attn_combine` | - | 1 | - | - | - | attn_combine | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3; 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3 |
| `attn_combine` | - | 2-8 | - | - | - | attn_combine | 4B mtp short; 9B mtp short; 27B mtp short; 35B mtp short |
| `attn_decode` | `grf256 -DHD=n -DHPG=n -DHS=n -DTPW=n -DSB=n` | 1 | - | - | - | attention | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3; 27B base host layer mtp short tp2 tp3 |
| `attn_decode` | `grf256 -DHD=n -DHPG=n -DHS=n -DTPW=n -DSB=n` | 2-8 | - | - | - | attention | 4B mtp short; 9B mtp short; 27B mtp short |
| `attn_dsplit` | `grf256 -DHD=n -DHPG=n -DHS=n -DTPW=n -DSB=n -DHG=n -DNSUB=n -DTR=n -DROWVMT` | 1 | - | - | - | attention | 35B base layer mtp short tp2 tp3 |
| `attn_dsplit` | `grf256 -DHD=n -DHPG=n -DHS=n -DTPW=n -DSB=n -DHG=n -DNSUB=n -DTR=n -DROWVMT` | 2-8 | - | - | - | attention | 35B mtp short |
| `attn_gate_quant` | - | 1 | - | attn gate+quant | - | attn_gate_quant | 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3 |
| `attn_gate_quant` | - | 2-8 | - | attn gate+quant | - | attn_gate_quant | 27B mtp short; 35B mtp short |
| `attn_gate_quant` | - | chunk | - | attn gate+quant | - | attn_gate_quant | 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3 |
| `attn_gate_quant` | `-DH8F=n` | 1 | - | quant+f32 copy, attn gate+quant | - | attn_gate_quant | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3 |
| `attn_gate_quant` | `-DH8F=n` | 2-8 | - | quant+f32 copy, attn gate+quant | - | attn_gate_quant | 4B mtp short; 9B mtp short |
| `attn_gate_quant` | `-DH8F=n` | chunk | - | quant+f32 copy, attn gate+quant | - | attn_gate_quant | 4B base host layer mtp tp2 tp3; 9B base host layer mtp tp2 tp3 |
| `attn_prep` | `-DSB=n` | 1 | - | q/k norm+rope+kv store | - | attn_prep | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3; 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3 |
| `attn_prep` | `-DSB=n` | 2-8 | - | q/k norm+rope+kv store | - | attn_prep | 4B mtp short; 9B mtp short; 27B mtp short; 35B mtp short |
| `attn_prep` | `-DSB=n` | chunk | - | q/k norm+rope+kv store | - | attn_prep | 4B base host layer mtp tp2 tp3; 9B base host layer mtp tp2 tp3; 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3 |
| `attn_q8` | `grf256 -DHPG=n -DHS=n -DTPW=n -DSB=n` | chunk | - | - | - | attention | 4B base host layer mtp tp2 tp3; 9B base host layer mtp tp2 tp3; 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3 |
| `conv_state_update` | - | 2-8 | - | - | - | conv_state_update | 4B mtp short; 9B mtp short; 27B mtp short; 35B mtp short |
| `conv_state_update` | - | chunk | - | - | - | conv_state_update | 4B base host layer mtp tp2 tp3; 9B base host layer mtp tp2 tp3; 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3 |
| `decode_w8_sb` | - | chunk | host | - | IQ2_S IQ2_XS IQ2_XXS IQ3_S IQ3_XXS IQ4_XS Q4_K Q5_K Q6_K Q8_0 | decode_w8 | 9B host; 27B host |
| `decode_w8_sb` | - | chunk | w8 jit | - | IQ1_M IQ2_S IQ2_XS IQ2_XXS IQ3_S IQ3_XXS IQ4_XS Q2_K Q4_K Q5_K Q6_K Q8_0 | decode_w8 | 9B base host layer mtp tp2 tp3; 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3; Ink base |
| `decode_w8_sb` | - | load | - | - | Q4_K Q5_K Q6_K | decode_w8 | 4B base host layer mtp short tp2 tp3; Ink base short |
| `decode_w8_sb` | - | load | host | - | Q6_K | decode_w8 | 9B host |
| `decode_w8_sb` | - | load | w8 jit | - | Q6_K | decode_w8 | 9B base layer mtp short tp2 tp3 |
| `decode_w8_sl` | `-DSL_NSB=n` | chunk | w8 jit | - | IQ3_S IQ3_XXS IQ4_XS Q8_0 | decode_w8 | 9B tp3; 27B tp2 tp3; 35B base layer mtp tp2 tp3 |
| `decode_w8_sl` | `-DSL_NSB=n` | load | - | - | Q6_K | decode_w8 | Ink base short |
| `decode_w8_vb` | - | chunk | host | - | Q8_0 | decode_w8 | 9B host |
| `decode_w8_vb` | - | chunk | w8 jit | - | Q8_0 | decode_w8 | 9B base host layer mtp tp2; 35B base layer mtp tp2 tp3; Ink base |
| `gated_norm_quant` | - | 1 | - | gated norm+quant | - | gated_norm_quant | 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3 |
| `gated_norm_quant` | - | 2-8 | - | gated norm+quant | - | gated_norm_quant | 27B mtp short; 35B mtp short |
| `gated_norm_quant` | - | chunk | - | gated norm+quant | - | gated_norm_quant | 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3 |
| `gated_norm_quant` | `-DH8F=n` | 1 | - | quant+f32 copy, gated norm+quant | - | gated_norm_quant | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3 |
| `gated_norm_quant` | `-DH8F=n` | 2-8 | - | quant+f32 copy, gated norm+quant | - | gated_norm_quant | 4B mtp short; 9B mtp short |
| `gated_norm_quant` | `-DH8F=n` | chunk | - | quant+f32 copy, gated norm+quant | - | gated_norm_quant | 4B base host layer mtp tp2 tp3; 9B base host layer mtp tp2 tp3 |
| `gdn_conv` | `-DCT=n -DCONV_UPD` | 1 | - | conv+state update | - | gdn_conv | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3; 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3 |
| `gdn_conv` | `-DCT=n` | 2-8 | - | - | - | gdn_conv | 4B mtp short; 9B mtp short; 27B mtp short; 35B mtp short |
| `gdn_conv` | `-DCT=n` | chunk | - | - | - | gdn_conv | 4B base host layer mtp tp2 tp3; 9B base host layer mtp tp2 tp3; 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3 |
| `gdn_gates` | - | 2-8 | - | - | - | gdn_gates | 4B mtp; 9B mtp; 27B mtp; 35B mtp |
| `gdn_recur` | `grf256 -DGQ=n -DFUSE_GATES` | 1 | - | recurrence+gates | - | gdn_recur | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3; 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3 |
| `gdn_recur` | `grf256 -DGQ=n -DFUSE_GATES` | 2-8 | - | recurrence+gates | - | gdn_recur | 4B mtp short; 9B mtp short; 27B mtp short; 35B mtp short |
| `gdn_recur` | `grf256 -DGQ=n -DFUSE_GATES` | chunk | - | recurrence+gates | - | gdn_recur | 4B base host layer mtp tp2 tp3; 9B base host layer mtp tp2 tp3; 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3 |
| `gdn_recur` | `grf256 -DGQ=n` | 2-8 | - | - | - | gdn_recur | 4B mtp; 9B mtp; 27B mtp; 35B mtp |
| `gemm` | `grf256 -DQTYPE=n -DACT32` | chunk | host | - | IQ2_S IQ2_XS IQ2_XXS IQ3_S IQ3_XXS IQ4_XS Q4_K Q8_0 | gemm_ab gemm_down gemm_gate gemm_k gemm_qkv gemm_ssm_out gemm_up gemm_v gemm_z | 4B host; 9B host; 27B host |
| `gemm` | `grf256 -DQTYPE=n -DACT32` | chunk | w8 jit | - | IQ1_M IQ2_S IQ2_XS IQ2_XXS IQ3_S IQ3_XXS IQ4_XS Q2_K Q4_K Q6_K Q8_0 | gemm_ab gemm_down gemm_down_sh gemm_gate gemm_gate_sh gemm_k gemm_o gemm_q gemm_qkv gemm_router gemm_ssm_out gemm_up gemm_up_sh gemm_v gemm_z ink_ctrl_down ink_gate ink_k ink_kcp ink_o ink_q ink_rel ink_v ink_vcp mtp_eh_proj | 9B base host layer mtp tp2 tp3; 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3; Ink base |
| `gemm` | `grf256 -DQTYPE=n -DACT32` | chunk | w8 resident | - | Q5_K Q6_K Q8_0 | gemm_ab ink_gate ink_k ink_q ink_rel | 4B base host layer mtp tp2 tp3; Ink base |
| `gemm` | `grf256 -DQTYPE=n -DACT32` | load | - | - | - | self_check | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3; 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3; Ink base short |
| `gemm` | `grf256 -DQTYPE=n` | chunk | host | - | Q4_K Q5_K Q6_K Q8_0 | gemm_down gemm_gate gemm_k gemm_o gemm_q gemm_qkv gemm_ssm_out gemm_up gemm_v gemm_z | 4B host; 9B host |
| `gemm` | `grf256 -DQTYPE=n` | chunk | w8 jit | - | Q4_K Q5_K Q6_K Q8_0 | gemm_down gemm_gate gemm_k gemm_o gemm_q gemm_qkv gemm_ssm_out gemm_up gemm_v gemm_z mtp_eh_proj | 9B base host layer mtp tp2 tp3 |
| `gemm` | `grf256 -DQTYPE=n` | chunk | w8 resident | - | Q4_K Q5_K Q6_K Q8_0 | gemm_down gemm_gate gemm_k gemm_o gemm_q gemm_qkv gemm_ssm_out gemm_up gemm_v gemm_z mtp_eh_proj | 4B base host layer mtp tp2 tp3 |
| `gemm` | `grf256 -DQTYPE=n` | load | host | - | Q6_K | h8_e2e | 4B host; 9B host |
| `gemm` | `grf256 -DQTYPE=n` | load | w8 jit | - | Q6_K | h8_e2e | 9B base layer mtp short tp2 tp3 |
| `gemm` | `grf256 -DQTYPE=n` | load | w8 resident | - | Q6_K | h8_e2e | 4B base layer mtp short tp2 tp3 |
| `gemv` | `-DGROWS=n` | 1 | host raw | - | IQ3_S Q4_K Q5_K Q8_0 | gemm_ab gemm_down gemm_gate gemm_k gemm_o gemm_qkv gemm_ssm_out gemm_up gemm_v gemm_z lm_head | 4B host; 9B host; 27B host |
| `gemv` | `-DGROWS=n` | 1 | raw | - | IQ3_S Q2_K Q4_K Q5_K Q8_0 | gemm_ab gemm_down gemm_down_sh gemm_gate gemm_k gemm_o gemm_q gemm_qkv gemm_ssm_out gemm_up gemm_v gemm_z ink_ctrl_down ink_gate ink_k ink_kcp ink_o ink_q ink_qkv_stack ink_rel ink_v ink_vcp lm_head mtp_eh_proj | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3; 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3; Ink base short |
| `gemv` | `-DGROWS=n` | 1 | raw | qkv+z | IQ3_S Q8_0 | gemm_qkvz | 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3 |
| `gemv` | `-DGROWS=n` | 1 | raw | router+shared gate/up | Q8_0 | gemm_router_sh | 35B base layer mtp short tp2 tp3 |
| `gemv_mt` | `grf256 -DGROWS=n -DTB=n` | 2-8 | raw | - | Q2_K Q4_K Q5_K Q8_0 | gemm_ab gemm_down gemm_down_sh gemm_gate gemm_gate_sh gemm_k gemm_o gemm_q gemm_qkv gemm_router gemm_ssm_out gemm_up gemm_up_sh gemm_v gemm_z ink_ctrl_down ink_gate ink_k ink_kcp ink_o ink_q ink_rel ink_v ink_vcp lm_head mtp_eh_proj | 4B mtp short; 9B mtp short; 27B mtp short; 35B mtp short; Ink short |
| `gemv_mt_v` | `grf256 -DGROWS=n -DTB=n` | 2-8 | raw | - | IQ1_M IQ2_S IQ2_XS IQ2_XXS IQ3_S IQ3_XXS IQ4_XS Q6_K | gemm_down gemm_gate gemm_k gemm_o gemm_q gemm_qkv gemm_ssm_out gemm_up gemm_v gemm_z ink_gate ink_k ink_q ink_rel lm_head mtp_eh_proj | 4B mtp short; 9B mtp short; 27B mtp short; 35B mtp; Ink short |
| `gemv_v` | `-DGROWS=n -DSOFTCAP=3.000000000e+01f` | 1 | raw | - | Q6_K | lm_head | Ink base short |
| `gemv_v` | `-DGROWS=n` | 1 | host raw | - | IQ2_S IQ2_XS IQ2_XXS IQ3_XXS IQ4_XS Q5_K Q6_K | gemm_down gemm_gate gemm_k gemm_q gemm_qkv gemm_ssm_out gemm_up gemm_v gemm_z lm_head | 4B host; 9B host; 27B host |
| `gemv_v` | `-DGROWS=n` | 1 | host raw | qkv+z | IQ4_XS | gemm_qkvz | 27B host |
| `gemv_v` | `-DGROWS=n` | 1 | raw | - | IQ1_M IQ2_S IQ2_XS IQ2_XXS IQ3_XXS IQ4_XS Q5_K Q6_K | gemm_down gemm_gate gemm_k gemm_o gemm_q gemm_qkv gemm_ssm_out gemm_up gemm_v gemm_z ink_gate ink_k ink_q ink_qkv_stack ink_rel lm_head mtp_eh_proj | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3; 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3; Ink base short |
| `gemv_v` | `-DGROWS=n` | 1 | raw | qkv+z | IQ3_XXS IQ4_XS | gemm_qkvz | 27B base host layer mtp short tp2 tp3 |
| `h8r_act` | `grf256 -DHB=n -DH8F=n` | chunk | host | quant+f32 copy | Q4_K Q6_K Q8_0 | h8_act | 4B host; 9B host |
| `h8r_act` | `grf256 -DHB=n -DH8F=n` | chunk | host | swiglu+rot, quant+f32 copy | Q6_K | h8_act | 4B host; 9B host |
| `h8r_act` | `grf256 -DHB=n -DH8F=n` | chunk | w8 jit | quant+f32 copy | Q4_K Q5_K Q6_K Q8_0 | h8_act | 9B base host layer mtp tp2 tp3 |
| `h8r_act` | `grf256 -DHB=n -DH8F=n` | chunk | w8 jit | swiglu+rot, quant+f32 copy | Q4_K Q5_K Q6_K | h8_act | 9B base host layer mtp tp2 tp3 |
| `h8r_act` | `grf256 -DHB=n -DH8F=n` | chunk | w8 resident | quant+f32 copy | Q4_K Q8_0 | h8_act | 4B base host layer mtp tp2 tp3 |
| `h8r_act` | `grf256 -DHB=n -DH8F=n` | chunk | w8 resident | swiglu+rot, quant+f32 copy | Q4_K Q5_K Q6_K | h8_act | 4B base host layer mtp tp2 tp3 |
| `h8r_act` | `grf256 -DHB=n -DH8F=n` | load | host | quant+f32 copy | Q6_K | h8_act | 4B host; 9B host |
| `h8r_act` | `grf256 -DHB=n -DH8F=n` | load | w8 jit | quant+f32 copy | Q6_K | h8_act | 9B base layer mtp short tp2 tp3 |
| `h8r_act` | `grf256 -DHB=n -DH8F=n` | load | w8 resident | quant+f32 copy | Q6_K | h8_act | 4B base layer mtp short tp2 tp3 |
| `h8r_colmax` | `grf256 -DHB=n -DSRC=n` | chunk | host | - | Q4_K Q5_K Q6_K Q8_0 | h8_colmax | 4B host; 9B host |
| `h8r_colmax` | `grf256 -DHB=n -DSRC=n` | chunk | host | swiglu+rot | Q6_K | h8_colmax | 4B host; 9B host |
| `h8r_colmax` | `grf256 -DHB=n -DSRC=n` | chunk | w8 jit | - | Q4_K Q5_K Q6_K Q8_0 | h8_colmax | 9B base host layer mtp tp2 tp3 |
| `h8r_colmax` | `grf256 -DHB=n -DSRC=n` | chunk | w8 jit | swiglu+rot | Q4_K Q5_K Q6_K | h8_colmax | 9B base host layer mtp tp2 tp3 |
| `h8r_colmax` | `grf256 -DHB=n -DSRC=n` | chunk | w8 resident | - | Q4_K Q5_K Q6_K Q8_0 | h8_colmax | 4B base host layer mtp tp2 tp3 |
| `h8r_colmax` | `grf256 -DHB=n -DSRC=n` | chunk | w8 resident | swiglu+rot | Q4_K Q5_K Q6_K | h8_colmax | 4B base host layer mtp tp2 tp3 |
| `h8r_colmax` | `grf256 -DHB=n -DSRC=n` | load | host | - | Q6_K | h8_colmax | 4B host; 9B host |
| `h8r_colmax` | `grf256 -DHB=n -DSRC=n` | load | w8 jit | - | Q6_K | h8_colmax | 9B base layer mtp short tp2 tp3 |
| `h8r_colmax` | `grf256 -DHB=n -DSRC=n` | load | w8 resident | - | Q6_K | h8_colmax | 4B base layer mtp short tp2 tp3 |
| `h8r_decode` | `grf256 -DH8RDEC -DH8DR=n -DHB=n -DSRC=n` | chunk | w8 jit | - | Q6_K Q8_0 | h8_decode | 9B tp2 tp3 |
| `h8r_decode` | `grf256 -DH8RDEC -DH8DR=n -DHB=n -DSRC=n` | chunk | w8 jit | swiglu+rot | Q6_K | h8_decode | 9B tp2 tp3 |
| `h8r_weights` | `grf256 -DHB=n -DSRC=n` | chunk | host | - | Q4_K Q5_K Q6_K Q8_0 | h8_weights | 4B host; 9B host |
| `h8r_weights` | `grf256 -DHB=n -DSRC=n` | chunk | host | swiglu+rot | Q6_K | h8_weights | 4B host; 9B host |
| `h8r_weights` | `grf256 -DHB=n -DSRC=n` | chunk | w8 jit | - | Q4_K Q5_K Q6_K Q8_0 | h8_weights | 9B base host layer mtp tp2 tp3 |
| `h8r_weights` | `grf256 -DHB=n -DSRC=n` | chunk | w8 jit | swiglu+rot | Q4_K Q5_K Q6_K | h8_weights | 9B base host layer mtp tp2 tp3 |
| `h8r_weights` | `grf256 -DHB=n -DSRC=n` | chunk | w8 resident | - | Q4_K Q5_K Q6_K Q8_0 | h8_weights | 4B base host layer mtp tp2 tp3 |
| `h8r_weights` | `grf256 -DHB=n -DSRC=n` | chunk | w8 resident | swiglu+rot | Q4_K Q5_K Q6_K | h8_weights | 4B base host layer mtp tp2 tp3 |
| `h8r_weights` | `grf256 -DHB=n -DSRC=n` | load | host | - | Q6_K | h8_weights | 4B host; 9B host |
| `h8r_weights` | `grf256 -DHB=n -DSRC=n` | load | w8 jit | - | Q6_K | h8_weights | 9B base layer mtp short tp2 tp3 |
| `h8r_weights` | `grf256 -DHB=n -DSRC=n` | load | w8 resident | - | Q6_K | h8_weights | 4B base layer mtp short tp2 tp3 |
| `ink_add` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | - | - | ink_add | Ink short |
| `ink_add` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | - | - | ink_add | Ink base |
| `ink_attn` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | - | - | ink_attn | Ink base short |
| `ink_attn` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | - | - | ink_attn | Ink short |
| `ink_attn` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | - | - | ink_attn | Ink base |
| `ink_conv` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | - | - | ink_conv | Ink short |
| `ink_conv` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | - | - | ink_conv | Ink base |
| `ink_conv_state` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | - | - | ink_conv_state | Ink short |
| `ink_conv_state` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | - | - | ink_conv_state | Ink base |
| `ink_conv1` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | - | - | ink_conv1 | Ink base short |
| `ink_ctrl_fused` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | controller | - | ink_ctrl_fused | Ink base short |
| `ink_ctrl_fused` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | controller | - | ink_ctrl_fused | Ink short |
| `ink_ctrl_fused` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | controller | - | ink_ctrl_fused | Ink base |
| `ink_mask_logits` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | - | - | ink_mask_logits | Ink base short |
| `ink_mix` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | - | - | ink_mix | Ink base short |
| `ink_mix` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | - | - | ink_mix | Ink short |
| `ink_mix` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | - | - | ink_mix | Ink base |
| `ink_mm_f32_sg` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | - | - | ink_router ink_sink | Ink base short |
| `ink_mm_f32_sg` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | - | - | ink_router ink_sink | Ink short |
| `ink_mm_f32_sg` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | - | - | ink_router ink_sink | Ink base |
| `ink_moe_out` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | - | - | ink_moe_out | Ink base short |
| `ink_rms2_quant` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | - | - | ink_rms2_quant | Ink base short |
| `ink_rms2_quant` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | - | - | ink_rms2_quant | Ink short |
| `ink_rms2_quant` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | - | - | ink_rms2_quant | Ink base |
| `ink_rmsnorm` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | - | - | ink_rmsnorm | Ink base short |
| `ink_rmsnorm` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | - | - | ink_rmsnorm | Ink short |
| `ink_rmsnorm` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | - | - | ink_rmsnorm | Ink base |
| `ink_rmsnorm_quant` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | - | - | ink_rmsnorm_quant | Ink base short |
| `ink_rmsnorm_quant` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | - | - | ink_rmsnorm_quant | Ink short |
| `ink_rmsnorm_quant` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | - | - | ink_rmsnorm_quant | Ink base |
| `ink_rows_kv` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 1 | - | - | - | ink_rows_kv | Ink base short |
| `ink_rows_kv` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | 2-8 | - | - | - | ink_rows_kv | Ink short |
| `ink_rows_kv` | `-DINK_LS=n -DINK_D=n -DINK_EXT=n` | chunk | - | - | - | ink_rows_kv | Ink base |
| `moe_combine` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | 2-8 | - | - | - | moe_combine | Ink short |
| `moe_combine` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | chunk | - | - | - | moe_combine | Ink base |
| `moe_combine` | `grf256 -DNE=n -DNU=n -DTM=n` | 1 | - | - | - | moe_combine | 35B base layer mtp short |
| `moe_combine` | `grf256 -DNE=n -DNU=n -DTM=n` | 2-8 | - | - | - | moe_combine | 35B mtp short |
| `moe_combine` | `grf256 -DNE=n -DNU=n -DTM=n` | chunk | - | - | - | moe_combine | 35B base layer mtp |
| `moe_down_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE -DFV=n` | 1 | - | - | IQ4_NL NVFP4 Q5_0 | moe_down | Ink base short |
| `moe_down_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE -DFV=n` | 2-8 | - | - | IQ4_NL NVFP4 Q5_0 | moe_down | Ink short |
| `moe_down_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE -DFV=n` | chunk | - | - | IQ4_NL NVFP4 Q5_0 | moe_down | Ink base |
| `moe_down_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE` | 1 | - | - | IQ4_XS Q4_K Q6_K | moe_down | 35B base layer mtp short tp2 tp3 |
| `moe_down_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE` | 2-8 | - | - | IQ4_XS Q4_K Q6_K | moe_down | 35B mtp short |
| `moe_down_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE` | chunk | - | - | IQ4_XS Q4_K Q6_K | moe_down | 35B mtp tp2 tp3 |
| `moe_gateup_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE -DASCT` | 2-8 | - | - | IQ3_S IQ4_XS Q3_K | moe_gateup | 35B mtp short |
| `moe_gateup_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE -DASCT` | chunk | - | - | IQ3_S IQ4_XS Q3_K | moe_gateup | 35B mtp tp2 tp3 |
| `moe_gateup_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE -DHLD=n -DMOE_SITU -DSITU_B=4.0f -DSITU_LB=25.0f -DASCT` | 2-8 | - | - | IQ4_XS NVFP4 Q4_K Q5_K | moe_gateup | Ink short |
| `moe_gateup_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE -DHLD=n -DMOE_SITU -DSITU_B=4.0f -DSITU_LB=25.0f -DASCT` | chunk | - | - | IQ4_XS NVFP4 Q4_K Q5_K | moe_gateup | Ink base |
| `moe_gateup_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE -DHLD=n -DMOE_SITU -DSITU_B=4.0f -DSITU_LB=25.0f` | 1 | - | - | IQ4_XS NVFP4 Q4_K Q5_K | moe_gateup | Ink base short |
| `moe_gateup_q` | `-DMOE_Q -DGROWS=n -DTM=n -DXCACHE` | 1 | - | - | IQ3_S IQ4_XS Q3_K | moe_gateup | 35B base layer mtp short tp2 tp3 |
| `moe_quant` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | 1 | - | - | - | moe_quant | Ink base short |
| `moe_quant` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | 2-8 | - | - | - | moe_quant | Ink short |
| `moe_quant` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | chunk | - | - | - | moe_quant | Ink base |
| `moe_quant` | `grf256 -DNE=n -DNU=n -DTM=n` | 1 | - | - | - | moe_quant | 35B base layer mtp short tp2 tp3 |
| `moe_quant` | `grf256 -DNE=n -DNU=n -DTM=n` | 2-8 | - | - | - | moe_quant | 35B mtp short |
| `moe_quant` | `grf256 -DNE=n -DNU=n -DTM=n` | chunk | - | - | - | moe_quant | 35B base layer mtp tp2 tp3 |
| `moe_rg` | `grf256 -DMOE_RG -DXCACHE` | chunk | - | decode+expert gemm | IQ3_S IQ4_NL IQ4_XS NVFP4 Q6_K | moe_down moe_gateup | 35B base layer mtp tp2 tp3; Ink base |
| `moe_route` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | 2-8 | - | - | - | moe_route | Ink short |
| `moe_route` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | chunk | - | - | - | moe_route | Ink base |
| `moe_route` | `grf256 -DNE=n -DNU=n -DTM=n` | 2-8 | - | - | - | moe_route | 35B mtp short |
| `moe_route` | `grf256 -DNE=n -DNU=n -DTM=n` | chunk | - | - | - | moe_route | 35B base layer mtp tp2 tp3 |
| `moe_route1` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | 1 | - | - | - | moe_route | Ink base short |
| `moe_route1` | `grf256 -DNE=n -DNU=n -DTM=n` | 1 | - | - | - | moe_route | 35B base layer mtp short tp2 tp3 |
| `moe_scan` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | 2-8 | - | - | - | moe_scan | Ink short |
| `moe_scan` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | chunk | - | - | - | moe_scan | Ink base |
| `moe_scan` | `grf256 -DNE=n -DNU=n -DTM=n` | 2-8 | - | - | - | moe_scan | 35B mtp short |
| `moe_scan` | `grf256 -DNE=n -DNU=n -DTM=n` | chunk | - | - | - | moe_scan | 35B base layer mtp tp2 tp3 |
| `moe_scatter` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | 2-8 | - | - | - | moe_scatter | Ink short |
| `moe_scatter` | `grf256 -DNE=n -DNU=n -DTM=n -DFV=n -DROUTE_SIG -DESCALE -DRSCALE=8.0f` | chunk | - | - | - | moe_scatter | Ink base |
| `moe_scatter` | `grf256 -DNE=n -DNU=n -DTM=n` | 2-8 | - | - | - | moe_scatter | 35B mtp short |
| `moe_scatter` | `grf256 -DNE=n -DNU=n -DTM=n` | chunk | - | - | - | moe_scatter | 35B base layer mtp |
| `mtp_concat` | - | 1 | - | - | - | mtp_concat | 4B mtp; 9B mtp; 27B mtp; 35B mtp |
| `mtp_concat` | - | 2-8 | - | - | - | mtp_concat | 4B mtp; 9B mtp; 27B mtp; 35B mtp |
| `mtp_concat` | - | chunk | - | - | - | mtp_concat | 4B mtp; 9B mtp; 27B mtp; 35B mtp |
| `rd_block` | - | 1 | - | - | - | read_gbs | 4B base host short; 9B base host short; 27B base host short; 35B base short; Ink base short |
| `rd_block` | - | 2-8 | - | - | - | read_gbs | 4B mtp; 9B mtp; 27B mtp; 35B mtp |
| `rd_strip` | - | 1 | - | - | - | read_gbs | 4B base host short; 9B base host short; 27B base host short; 35B base short; Ink base short |
| `rd_strip` | - | 2-8 | - | - | - | read_gbs | 4B mtp; 9B mtp; 27B mtp; 35B mtp |
| `rmsnorm_h8r` | `grf256 -DH8NORM -DHB=n -DH8N_K=n` | chunk | - | norm+rot | - | rmsnorm_h8r | 4B base host layer mtp tp2 tp3 |
| `rmsnorm_quant` | - | 1 | - | norm+quant | - | mtp_concat rmsnorm_quant | 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3; Ink base short |
| `rmsnorm_quant` | - | 2-8 | - | norm+quant | - | mtp_concat rmsnorm_quant | 27B mtp short; 35B mtp short |
| `rmsnorm_quant` | - | chunk | - | norm+quant | - | mtp_concat rmsnorm_quant | 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3 |
| `rmsnorm_quant` | `grf256 -DH8F=n` | 1 | - | quant+f32 copy, norm+quant | - | mtp_concat rmsnorm_quant | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3 |
| `rmsnorm_quant` | `grf256 -DH8F=n` | 2-8 | - | quant+f32 copy, norm+quant | - | mtp_concat rmsnorm_quant | 4B mtp short; 9B mtp short |
| `rmsnorm_quant` | `grf256 -DH8F=n` | chunk | - | quant+f32 copy, norm+quant | - | mtp_concat rmsnorm_quant | 4B mtp; 9B base host layer mtp tp2 tp3 |
| `rmsnorm_quant` | `grf256 -DH8F=n` | load | - | quant+f32 copy, norm+quant | - | rmsnorm_quant | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3 |
| `swiglu_quant` | - | 1 | - | swiglu+quant | - | mtp_concat quant swiglu_quant | 27B base host layer mtp short tp2 tp3; 35B base layer mtp short tp2 tp3; Ink base short |
| `swiglu_quant` | - | 2-8 | - | swiglu+quant | - | mtp_concat quant swiglu_quant | 27B mtp short; 35B mtp short; Ink short |
| `swiglu_quant` | - | chunk | - | swiglu+quant | - | mtp_concat quant swiglu_quant | 27B base host layer mtp tp2 tp3; 35B base layer mtp tp2 tp3; Ink base |
| `swiglu_quant` | `-DH8F=n` | 1 | - | quant+f32 copy, swiglu+quant | - | mtp_concat swiglu_quant | 4B base host layer mtp short tp2 tp3; 9B base host layer mtp short tp2 tp3 |
| `swiglu_quant` | `-DH8F=n` | 2-8 | - | quant+f32 copy, swiglu+quant | - | mtp_concat swiglu_quant | 4B mtp short; 9B mtp short |
| `swiglu_quant` | `-DH8F=n` | chunk | - | quant+f32 copy, swiglu+quant | - | mtp_concat | 4B mtp; 9B mtp |
| `tp_add` | - | 1 | - | - | - | tp_add | 4B tp2 tp3; 9B tp2 tp3; 27B tp2 tp3; 35B tp2 tp3 |
| `tp_half` | - | chunk | - | - | - | tp_sum | 4B tp2 tp3; 9B tp2 tp3; 27B tp2 tp3; 35B tp2 tp3 |
| `tp_moe_combine` | - | 1 | - | - | - | moe_combine | 35B tp2 tp3 |
| `tp_moe_combine` | - | chunk | - | - | - | moe_combine | 35B tp2 tp3 |
| `tp_moe_mask` | - | chunk | - | - | - | moe_route | 35B tp2 tp3 |
| `tp_moe_scatter` | - | chunk | - | - | - | moe_scatter | 35B tp2 tp3 |
| `tp_moe_tiles1` | - | 1 | - | - | - | moe_route | 35B tp2 tp3 |
| `tp_pack` | - | 1 | - | - | - | tp_pack | 4B tp3; 9B tp3; 27B tp3; 35B tp3 |
| `tp_pack` | - | chunk | - | - | - | tp_pack | 4B tp3; 9B tp3; 27B tp3; 35B tp3 |
| `tp_rs_add_r` | - | chunk | - | - | - | tp_sum | 4B tp2 tp3; 9B tp2 tp3; 27B tp2 tp3; 35B tp2 tp3 |
| `tp_rs_sum` | - | chunk | - | - | - | tp_sum | 4B tp2 tp3; 9B tp2 tp3; 27B tp2 tp3; 35B tp2 tp3 |

**Forms missing, a faster one measured elsewhere** (checked against the code 2026-11-27):

| Shape | Ours | Faster, measured | Source |
|---|---|---|---|
| MoE experts in prompt chunks | Inkling `moe_down` 73.8 ms a visit (125H) | oneDNN batched 6.9 ms | plan-2026-11-30.md section 1 |
| the 35B's experts | 23.4 / 13.1 ms | oneDNN 13.5 / 7.9 (3.5x with decode_w8) | plan section 1 |
| thin GEMM, Inkling's controller 512 x 64 x 2304 | 10.2 ms | oneDNN 0.06 ms | plan section 1; `fix/ink-ctrl-gemm` unmerged |
| NVFP4 experts (unsloth's Inkling) | B70 `moe_down` 2,275 ms | the same model as IQ4_NL: 238 ms | hibagus, #92 (2026-11-24) |
| dense int8 prompt GEMM, DP4a iGPU | H8R 8.7 TOPS, W8 8.1 | oneDNN 12.1 TOPS (per-row scales; its kernel at that shape: unknown) | before-after.md, 2026-11-26 |
| one-token Q4_K GEMV, 64-EU Arrow Lake-S | 25-28 GB/s | oneDNN u4 GEMV 60 GB/s | jac2424, #97 (2026-11-26) |
| one-token Q5_K GEMV, 125H | 58.6-61 GB/s (gemv_bench 69-70) | Q6_K's path 67-68.5 (bench 76.4) | before-after.md, 2026-11-26 |
| 8-lane XMX prompt GEMM, A770 | 27B 4K ~1,773 tok/s | target 2,200 | QUEUE.md; before-after.md |
| f16 GEMM on DP4a iGPUs | none fast | oneDNN f16 3.6 TOPS at 512 x 2560 -> 9216 (125H, 2026-11-26) | the Researcher's run; not yet in before-after |
| the 27B on the fly, 125H | gate / up / down 1.55-1.6x, qkv 1.56x behind | oneDNN 3.14.2 | plan section 1 |
| Flash Next on the 125H | dense 1.7-2.1x, experts 5.7-7x behind | oneDNN 3.14.2 | plan section 1 |
| A770 int8 GEMM | ~65 % of peak | oneDNN 72-79 % | QUEUE.md |
| B60 under Windows | 157.6 TOPS | oneDNN 170-183 | QUEUE.md |
| B70 16-lane XMX GEMM | xw_peak's best 232 TOPS (`L128` not measured in the model) | oneDNN 233-237 | QUEUE.md; before-after.md |
| 8-lane XMX expert tiles | 1.9x / 3.2x slower than our own DP4a | none measured yet: the 8-lane XMX tile is slower than DP4a, so DP4a is the bar | QUEUE.md |
| Arc 140T prompt GEMM | ~33 % of its XMX int8 peak | none measured yet (no oneDNN run on a 140T): the peak is the bar | CONTRIBUTING.md |

## B. Every guard and invariant

Two kinds of row:
- **guard:** a check that refuses or falls back, proven by a test that triggers it;
- **invariant:** a rule the code must keep, proven by a hash that shows it holds.

The code side is read from the choice functions' conditions (matmul1, gemv_gu, gemv_rows, h8_takes /
h8_igpu_rows_ok, layer_moe's branch order, p140_take, the xe2 defaults per kind), not their comments. The history side
comes from CHANGELOG.md, docs/before-after.md, docs/audit-*.md and BUGS.md. The two are merged with each row's source.

### The code side (the Developer, 2026-11-28, read at main 6fb85d7)

Line numbers are main 6fb85d7's; since 3b102cd, engine.h's lines after 255 are one higher. Read from the code's conditions, not the comments. The findings marked "checked" were re-read by hand, and the Researcher re-read findings 1 and 4. "golden" is tests\gemm_golden.ps1 and its kind; "run_tests" is tests\run_tests.ps1. "(no comment)" means no comment says why. A row whose test is "none" is unproven: Phase 1 must not move it without adding one, or keep it unchanged and prove it by the golden hashes.

#### Findings (not changed: Phase 0 is inventory; each goes to a fix after Phase 0 or with its migration turn)

1. **XMX verdict cache per device only (checked).** engine.h:489-494 `xmx_by_dev[cl_.device_id]`; the speed check is skipped when `p140_` is on (engine_setup.h:857). In one server process on an Arc 140T, the first model loaded decides XMX for every later model on that GPU, its speed check included or not.
2. **"512-byte aligned" views are a comment only.** engine_stream.h:563 says so (rgu_rows rounded to 128 rows, engine_sessions.h:858); nothing checks it, and zrt's clCreateSubBuffer checks bounds only (clrt.h:1044-1052). The qz and Inkling stack views have no alignment rule.
3. **Staged W8 written without a size check.** matmul1 / h8_prepare decode into `stage_q_[role]` relying on `stage_need_*` / `ensure_stage` having covered every weight (engine.h:1722-1726; engine_alloc.h:207-209; engine_sessions.h:862-870).
4. **The spill warning is said once per name + options, not per GPU (checked).** clrt.h:1762-1765: the same kernel spilling on a second GPU kind of a layer split is not reported again.
5. **CL_CHECK exits with _Exit(1) (checked)**, no destructors, no marker (clrt.h:89-97): a CL error is neither a hang (3) nor a lost device (4).
6. engine.h:1729 (`stage_packed_` reset) is mis-indented under the `if` above it (cosmetic; the logic is what the braces say).

#### B.1 zero/engine.h

| kind | protects | file:line | test |
|---|---|---|---|
| guard | head softcap epilogue only for the head, softcap > 0, not `q2_nocap_` | 1309 | run_tests 4c |
| invariant | two-pass: the head's softcap applies to the sum of both passes | 1413-1414 | run_tests 4c |
| guard | fused two-pass GEMV (`q2_fuse_`) only at T <= 4 | 1416 | golden short / mtp |
| guard | no second pass when `!q2mm_ \|\| q2_off_`; the drafter forces one pass (engine_drafters.h:71-72) | 1420 | run_tests 4c |
| guard | two-pass with a fused epilogue (acc > 1) throws | 1421 | none |
| invariant | second pass reads aq_/asc_/absum_ at q2 offsets, skipped when fused (buffers x2: engine_alloc.h:10, 42, 44) | 1422-1430, 1395-1401 | run_tests 4c |
| guard | two-pass head over > 8 rows throws | 1434 | none |
| invariant | `q2mm_` decided once (q2_init): Gemma but E2B/E4B, Flash Next with ARC_FN_Q2, Qwen3 unless ARC_Q3_Q2=0 | 1387-1392 | run_tests 4c |
| invariant | `gemv_v_ok` type list (IQ3_S excluded: "measured" only) | 1320-1321 | none |
| guard | ARC_GVROWS outside 1-8 ignored; forced count only if <= N | 1330-1331, 1338 | none |
| invariant | Xe-HPC: 1 row unless ARC_GVROWS | 1335 | none |
| invariant | gemv_v rows: N > 4096 or K >= 8192 -> 4, but IQ4_XS / Q6_K / Q4_0 | 1343-1344 | golden base |
| guard | tiled IQ3_S never on gemv_v | 1451 | none |
| guard | K4 rule: !q2mm_, not the head; default Q5_K, N >= 8192, N >= 2K, iGPU, no XMX | 1461-1464 | none |
| guard | ARC_GV_SMALL: iGPU, no XMX, Xe-LPG <= 640 threads (or SMALL_ANY) | 1474-1476 | none |
| guard | GV_SMALL=2 widening needs gv_k4 != 0, !q2mm_, not the head | 1477-1478 | none |
| invariant | Alchemist (sg8 + XMX + discrete + unbatched): vk types 1 row | 1488 | none |
| guard | ARC_GEMV_ROWS / gv_small overrides only for !vk and N <= 65536 | 1492-1493 | none |
| invariant | default rows 1 / 2 / 4 / 2 by N, then xe2_gemv_rows | 1495 | golden base / short |
| invariant | host_gemv_mb: 0 on an iGPU, 4 MB discrete, per call | 1499-1502 | golden host |
| guard | gemv_gu: env, T == 1, !q2mm_, both raw, same type/N/K, not the head | 1518-1520 | golden fusion GEMV_GU=1 |
| guard | gemv_gu: iq3t must match | 1521 | none |
| guard | gemv_gu: host weights refused | 1522 | golden host |
| guard | gemv_gu: N % grows refused | 1528 | none |
| invariant | gemv_gu kernel / rows cached on g at first use, never re-evaluated | 1523-1534 | none |
| guard | gemv split-K: refused when off, ARC_GEMV_ROWS set, no thread count, N > 65536, MoE | 1556-1557 | none |
| guard | SPLITK_FORCE: power of two 2-8, <= steps | 1573 | none |
| guard | host-CPU GEMV (ARC_HOST_CPU) only with act32 | 1593 | none |
| invariant | one-token kernel cached per weight (g1k[q2_fuse_], g1rows, g1vk, g1srows, g1split) | 1602-1626 | none |
| guard | GV_PROBE never on vk types or the head | 1613 | none |
| guard | split-K GEMV excluded for fused two-pass, iq3t, probe, head, GEMV_DEEP | 1617-1619 | none |
| guard | fused two-pass GEMV with an epilogue throws | 1631 | none |
| guard | host pieces only if host, not fused, split 1, host_mb > 0, rows > host_mb; whole row groups | 1646-1648 | golden host |
| invariant | GEMV grid / WG 256 for gemv_v or split | 1660-1661 | none |
| invariant | 2-8 rows: vk adds IQ3_S (untiled) and Qwen3 Q8_0 fused | 1682 | golden mtp |
| invariant | 2-8 rows: Q6_K 1, T < 6 -> 4 / 2, then xe2_gemv_mt_rows | 1689-1697 | golden mtp |
| guard | 2-8 rows two-pass fuse only when q2_fuse_ && vk && a <= 1 | 1696 | run_tests 4c |
| guard | GEMM: ldc % 4 or c_off % 4 throws | 1714-1716 | none |
| guard | h8_takes tested first on the GEMM path | 1719 | golden option H8=0 |
| invariant | staged W8 re-decoded unless owner and not the H8 form | 1725-1730 | golden base |
| guard | x16 GEMM tried first, falls back when it returns false | 1732 | none |
| invariant | 140T preset tile choice (32 tokens / cg 2; 16 / cg 1; else xe2_cg_for) | 1735-1737, 1743 | xmx_check mode 1 |
| guard | GEMM shape Mp % tm, N % 16, K % 256 throws | 1745-1747 | none |
| invariant | ARC_GEMM_SPLIT: 0-4096, default 17; only a != 2 and K > 8192; pieces 2+ accumulate | 1751, 1754, 1759 | none |
| invariant | wgy doubles while it divides Mp / tm, up to ARC_GEMM_WGY 8 (no comment) | 1756 | none |
| guard | decode_w8: N % 16 throws | 1277 | none |
| invariant | decode_w8 WG halves until it divides N; vb needs lx >= 64 and Q8_0 / Q4_0 (no comment on 64) | 1286-1287 | none |
| guard | decode_w8_sl: no requant, nsb <= 16, N % 16, <= 32 KB, not IQ1_M / IQ1_S (no comment on IQ1 / 32 KB) | 1291-1292 | none |
| guard | short-prompt 8-row passes only when !keep_rows, !w8_resident, !stream_, n <= few, max_chunk >= 64 | 752-753 | golden short |
| invariant | Mp: >= 128 rounded to 128 (capped at max_chunk), else to 16 | 758, 802-805 | none |

#### B.2 zero/engine_h8.h

| kind | protects | file:line | test |
|---|---|---|---|
| guard | h8_init: (ARC_H8 \|\| p140_ \|\| h8_auto_ \|\| h8_dp4a_) && !q2mm_ && !npu | 84 | golden option |
| guard | f32 source only when H8R, not ARC_H8_INT, !hc, !gemma | 77, 85-88 | none |
| invariant | h8_block: largest power of two <= cap dividing K, >= 256 | 108-116 | none |
| guard | h8_ok: role 0-10, qtype Q8_0 / 100-103, K % 256, N % 16; H8R K / hb <= 32, K <= 130000 | 117-121 | none |
| guard | h8_takes: T > 8, !q2mm_, h8_ok, 140T take rule, resident or h8_igpu_rows_ok | 265-267 | golden base / option |
| invariant (A1) | resident W8 always takes H8 (rotated in place); the plain GEMM never reads a rotated copy | 266, 276-278 | h8_e2e_check |
| guard (spec_) | h8_dp4a_ without ARC_H8: not in verification passes for JIT weights | 285 | golden mtp |
| guard | 16-lane-XMX iGPU: H8 from Mp >= ARC_H8_MIN (256) | 286-288 | none |
| guard | h8_swi_ok: env, H8R, f32 copy, h8_takes(down) | 290-293 | golden fusion H8_SWI=0 |
| guard | h8_norm_env defaults: p140_ \|\| discrete non-Xe-HPC \|\| (h8_dp4a_ && hb <= 512) | 319-323 | golden fusion H8_NORM=0 |
| guard | rmsnorm_h8 falls back on any of 11 conditions (no share scope, wide rows, K / hb > 16, row > 60 KB ...) | 331-332 | golden fusion |
| invariant | h8_pre_live_ cleared by rmsnorm_quant / ~H8Share / any rotation; used only if Mp and K match | 255, 347, 464-465 | golden fusion H8_NORM=0 |
| guard | h8_out_env default only for discrete sg8 (Alchemist) | 358-361 | none |
| guard | h8_out_ok: rmsnorm_h8's row limits | 362-366 | none |
| invariant | rotation reuse only if K, M, act32 and the source match | 467-470 | golden fusion H8_REUSE=0 |
| guard | H8 GEMM without h8_act in its call throws (h8_input_check) | 456-459, 589, 610 | none |
| guard | h8_act with an f32 source without H8R throws | 485 | none |
| guard | h8_e2e_check A/B/A on layer 0's qkv; skipped for MoE / hc / gemma / ds41 / inkling, ARC_H8_E2E=0, dry | 393-455 (engine.h:605) | itself |
| invariant | e2e Restore keeps resident rotated entries | 427-441 | none |
| guard | h8_dec_ok: one-pass decode+rotate type list, hb >= 512, N % 16 (no comment on IQ1) | 193-202 | none |
| guard | h8_prepare: a resident weight asked for packed order throws | 498 | none |
| invariant (A1) | resident copy rotated once; p140_res_ keeps them unrotated; w8_part_auto erases the mark on a fresh decode | 497-506; engine_load.h:260 | h8_e2e_check (#80) |
| invariant | staged H8 re-decoded unless owner, H8 flag and packed order match | 514 | golden base |
| guard | H8SB scales buffer grown when small | 542-544 | none |
| invariant | h8_splits: 17 superblocks iGPU / 140T, 34 discrete; never with a == 2 (decides bits) | 567-572 | none |
| guard | ARC_H8_Q2 with fused SwiGLU throws | 579 | none |
| guard | H8R int32: a piece over 133,000 values throws | 593-598 | none |
| guard | packed operands not taken by gemm_xw8: throws | 617-618 | xw_check |
| guard | H8 GEMM shape throws | 625-626 | none |

#### B.3 zero/xe2/engine_xe2.h and the setup (engine.h)

| kind | protects | file:line | test |
|---|---|---|---|
| guard | h8_dp4a_wanted: not with ARC_H8, XMX, discrete, non-Xe-LPG, moe/gemma/hc/qwen3/ds41/inkling, !h8_row; resident yes, else gate Q4_K / Q5_K | xe2:60-68 | none |
| guard | h8_auto_wanted: XMX, 16 lanes, not moe/gemma/hc/qwen3 | xe2:69-72 | golden option |
| guard | p140_wanted: XMX (or test), sg8, resident rule, not moe/gemma/hc/qwen3, H8R | xe2:79-92 | xmx_check mode 1 |
| guard | p140_take: qtype 100-103, Mp >= 512 (768 res), Mp % 32, N % 32 (no comment on N % 32) | xe2:30-32 | xmx_check 140T |
| invariant | setup order q2_init -> p140_ -> h8_auto_ -> h8_dp4a_ -> h8_init -> p140_ &= h8_ | engine.h:455-465 | none |
| invariant | h8_cg_for: 4 groups with h8_auto_ on discrete when (N/16) % 4 == 0 | xe2:75-78 | xmx_check mode 1 |
| invariant | xehpc_on / xe2_g21 / t1280 / moe_grf: per engine | xe2:133-137, 150-162, 217-227 | none |
| invariant | xe2_cg_for only with xe2_ or ARC_GEMM_CG, (N/16) % cg | xe2:230-240 | xmx_check (cg <= 4) |
| invariant | xe2_gemv_rows / mt_rows / wg rules | xe2:309-352 | golden mtp (mt) |
| guard | 8-lane iGPU: XMX off unless FORCE / TRY / p140_ | engine.h:474-479 | none |
| guard | xmx_check mode 0 once per device_id; 2D retried without; sets xmx_ / slow. **Finding 1** | engine.h:487-531 | start-up |
| guard | xmx_check(1) for H8 keyed by device, qtype, cg, 140t, env, 2d; failure drops p140_ / h8_auto_ / h8_, keeps XMX | engine.h:536-551 | start-up |
| guard | p140_ / h8_auto_ without xmx_: off | engine.h:554-561 | none |
| invariant | H8 off after checks: h8f_off_ dropped, kernels rebuilt | engine.h:566-570 | none |
| guard | x16: wanted rule; x16_check once per device; shape refusals | xe2:823-847, 882, 911-1003 | x16_check |
| guard | xw_init / xw_check per key; fall back DPASW -> split barrier -> C_RND -> 2D -> off | engine_xw.h:1060-1198 | xw_check |
| guard | rgx_first_use: DP4a vs XMX copy byte compare per type key; differs or no room -> DP4a for that type, layer back to decode + GEMM | engine_moe_xw.h:78-127, 43-53; engine_moe.h:614-616 | itself |
| guard | rgx_kernel build failure -> DP4a (soft_fail) | engine_moe_xw.h:55-66 | none |

#### B.4 zero/engine_setup.h

| kind | protects | file:line | test |
|---|---|---|---|
| guard | gemm_self_check: Q8_0 64 x 1024 vs CPU; rel > 1e-3 -> warning + driver_warning, never a refusal; ARC_SELF_CHECK=0 skip, =2 fake | 612-694 (engine.h:472) | ARC_SELF_CHECK=2 |
| guard | xmx_check mode 0: qtypes x act32 x store / split / add / swiglu, bit for bit vs DP4a; MoE tiles | 735-780 | start-up |
| guard | MoE x8 failure on sg8: moe_x8_ok_ off, dense XMX kept | 781-813 | none |
| guard | mode 1 140T cases | 817-846 | start-up |
| guard | speed check: XMX must be 3 % faster on 512 x 17408 x 5120; skipped with FORCE / p140_ / TRY (**finding 1**) | 857-907 | start-up |
| guard | build exceptions -> DP4a; GpuWaitError rethrown | 909-916 | none |
| invariant | p140_res_ only with ARC_140T_RES, resident, not TP, discrete | 437-440 | none |

#### B.5 zero/engine_moe.h (layer_moe, line 8)

| kind | protects | file:line | test |
|---|---|---|---|
| invariant | q2on / q2f / q2g (fused per tile, two-pass on the GEMM) | 99-102 | run_tests 4c 26B |
| invariant | tile size rules; ARC_MOE_TM_SMALL in {0,1,2,4,8,16} | 87-94, 106 | none |
| invariant | max_tiles grid, extra WGs exit | 167 | none |
| guard | gemm_ok: moe_group_, Fp == F, moe_gemm_layer_ok, !tp_moe, !ds41 | 196 | run_tests 35B |
| guard | rg_ok: moe_std, tm 16, T >= rg_min or q2g, F % 64, layer ok, !ds41 (no comment on tm / F % 64 here) | 197 | run_tests 35B |
| invariant | gate/up order: moe_rg -> moe_gemm -> per tile gateup_q -> GSQ legacy | 212-219, 447-458 | run_tests 35B; golden 35B |
| invariant | down order: moe_rg -> moe_gemm (tm 16, T >= 1152) -> per tile down_q -> GSQ legacy (no comment on tm 16) | 215-216, 470-481 | same |
| guard | moe_gemm_part_ok: W8 type, not Q8_0 / Q4_0, variant < 2 | 527-533 | none |
| guard | moe_gemm_pad_on: discrete 16-lane, padded F, not gemma | 544-554 | none |
| guard | moe_rg_type list | 562-570 | none |
| guard | XCPU path: xc_, T == 1, !use_gemm, moe_std, xsplit, !q2g, (!tp_moe \|\| xsplit_tp) | 337 | none |
| guard | xsplit_tp off unless ARC_XSPLIT_TP=1 (F30) | 523-526 | none |
| guard | xcpu_layer_ok: discrete 16-lane default, moe_std, not gemma / inkling | xe2/engine_xcpu.h:50-55, 72-79 | none |
| invariant | xgpu: GPU tiles skipped when the CPU took every expert | 416, 452, 460, 475 | none |
| invariant | moe_rg two-pass then moe_gelu_mul | 597-629 | run_tests 4c 26B |
| invariant | moe_gemm sub-buffers per group (+kRawSlack, clamped) | 640-648 | none |

#### B.6 Layout contracts

| kind | protects | file:line | test |
|---|---|---|---|
| invariant | kRawSlack = 1024 bytes after the last superblock (GEMV reads past a row) | engine_load.h:98, 134, 178 | none |
| invariant | superblocks padded to 8 bytes | src/quants.h:857; engine_load.h:131 | test_quant_rows (CPU) |
| guard | IQ3_S tiling only K % 512, bp 112, nsb even, ARC_IQ3S_TILE; reported to w.iq3t | engine_load.h:132-133, 278 | none |
| invariant | host flag: up_host_ copied by every upload_padded caller | engine_load.h:137-142, 279; engine_load_gemma.h:142, 158, 254, 265, 270; engine_stream.h:574; engine_drafters.h:751; engine_inkling.h:273 | golden host |
| invariant | views copy their stack's iq3t / host by hand | engine_load.h:450-455; engine_stream.h:583; engine_inkling.h:290-291 | golden host |
| invariant | view size N * sbr + kRawSlack | engine_load.h:439; engine_stream.h:588; engine_inkling.h:277 | none |
| invariant | "512-byte aligned" views: comment only (**finding 2**) | engine_stream.h:563 | none |
| invariant | w8_bytes_q = N*K, w8_bytes_h = N/8 * K/256 * 320 | src/quants.h:832-833 | none |
| invariant | stage_need_q_/h_ max per role; ensure_stage for later weights; no size check at the write (**finding 3**) | engine_load.h:285-286, 311-316; engine_stream.h:584-585; engine_alloc.h:207-209; engine_sessions.h:862-870; engine.h:1722-1726 | none |
| invariant | H8SB scales sized stage_need_h_ / 8 + 64 | engine_h8.h:135 | none |

#### B.7 Process-wide statics tied to a device

None: no `static const` initializer calls zrt::integrated(), xe_lpg(), cl_.xmx or a device id (single-line and multi-line static lambdas in zero/*.h, zero/xe2/*.h, zero/mgpu/*.h). Caches keyed by device: xmx_by_dev (engine.h:488-489, **finding 1**), h8_known (537-538), x16 known (xe2:834-835), the xw maps (engine_xw.h:1060-1079), first_for_device (engine.h:1245-1246). Not keyed by device: the spill warning's set (clrt.h:1762-1765, **finding 4**), once_flag messages (xe2:154, engine_moe_xw.h:122).

#### B.8 Cache keys

| kind | protects | file:line | test |
|---|---|---|---|
| invariant | kernel binary = fnv1a(src, opts, device_id = CL name + driver); a damaged binary is recompiled | clrt.h:1643, 1412, 1708-1716 | none |
| invariant | warm marker = fnv1a(sources, model path, warm_key_extra (all ARC_*), split_key, mtp / dflash / drafts / max_chunk / device_id, "w3") | engine.h:994-1033 | none |
| invariant | split_key: "\|tp<n>\|fnv(tp.cl)" or "\|pp<n>" + each stage's range@gpu_id | tools/server_load.h:393-411 | none |

#### B.9 Failure paths

| kind | protects | file:line | test |
|---|---|---|---|
| guard | CL_CHECK: message + _Exit(1) (**finding 5**) | clrt.h:89-97 | none |
| guard | check_marker: a marker from this boot (< 12 h) -> _Exit(3); ARC_IGNORE_HANG warns | gpu_fail.h:161-196; clrt.h:1481 | gpu_guard |
| guard | gpu_fatal: Hang 3 / Lost 4 write the marker; ze_exit 0x70000001 -> 4, else 5 | gpu_fail.h:29, 218-241, 262-264 | gpu_guard |
| guard | ARC_GPU_WAIT_S >= 5 s; a longer wait -> Hang | clrt.h:371-383, 409-411 | gpu_guard |
| guard | per-kernel wait (ARC_ZE_WATCH / SYNC_EACH) -> Hang | clrt.h:1165-1169, 1320-1329 | none |
| guard | ARC_GPU_FAULT_TEST=hang / lost | clrt.h:398-404; gpu_fail.h:268-276 | manual |
| guard | launch: g == 0 none; g > 2^32 refused; failed append -> ze_exit | clrt.h:1251-1255, 1308-1313 | none |
| invariant | run(..., nullptr): suggestion shrunk until it divides; a given lws that does not divide -> -54 | clrt.h:1260-1280 | none |
| guard | spill warning once per name + opts (**finding 4**) | clrt.h:1753-1770 | none |
| guard | cancel: between chunks; per layer only when !w8_resident; layer-major start and after each layer (TP barrier); never inside matmul or generation | engine.h:745-766, 806-862; engine_sessions.h:221-238 | none |
| guard | ARC_FAIL_PASS: the N-th verify / batched pass throws after writing its state | engine_slots.h:386-391; tools/server_load.h:431 | tests/fail_inject.ps1 |

Not tested by any dedicated test: every logic_error / runtime_error shape refusal in matmul1 and h8_matmul.

### The history side

To come (the Researcher): each row's commit and the bug or audit that added it, from CHANGELOG.md, docs/before-after.md, docs/audit-*.md and BUGS.md, merged into the rows above.

## C. The golden test set

Per model, logits hashes (`runz --dump-logits`) with the shared layer on and off (`ARC_GEMM_SHARED`), equal bit for
bit. Each set is recorded with its build and driver version, since the f32 parts can move with a driver's compiler.
1. a prompt longer than one chunk, then 16 generated tokens;
2. a short prompt (under 8 tokens): the 2-8-row paths;
3. drafting on (MTP / DFlash2 / DSpark) where the model has it;
4. two slots batched against each alone;
5. the memory forms: host-memory weights forced, resident W8 against decoded per chunk;
6. the splits under `ARC_TP_SIM`: `--gpus 2` and `3` (its own hash: not bit-identical to one GPU) and `--split layer`
   (bit-identical to one GPU);
7. the switches that change kernels (`ARC_H8=0`, `ARC_BATCH`, `ARC_GEMV_GU`, ...), one hash each.

The DPAS emulations (`ARC_MOE_RG_XEMU`, `ARC_GDN_XEMU`, the attention one) cover moe_rg, DeltaNet and attention on
the laptop. `gemm_x8`, `gemm_xw` and `gemm_x16` stay contributor-only.

**Generation at every depth** (the user, 2026-11-28: generation covered, not prefill only, for short, medium and long
prompts up to the prompt size limit). Kind 1 checks generation after ~1,100 tokens only, but decode changes route with
depth (the attention past short contexts splits over the keys, design 3.7a) and the KV cache grows. So a **depth**
kind:
- prompt lengths: short (4), medium (512, 4,096), long (16,384, 32,768, 65,536, 131,072) and **the limit**: the
  context `--ctx auto` gives the model on that GPU, minus the generated tokens. Each length up to the limit only;
- the prompt is the fixed text repeated to the length (real text, so the model keeps writing), then **64 generated
  tokens**, all hashed (`--dump-logits` holds every step);
- base path only, not the fusion and option matrix, since it's long: the long lengths run under a `-Long` switch.
  On the laptop the 27B's 163,840 alone is about half an hour;
- the splits (tp2 / tp3 / layer) at medium and at their own limit.
**The harness:** `tests/gemm_golden.ps1` (merged 2026-11-28, 5d63cc2) runs these kinds through `bench\gpu_guard.ps1`
and writes `results.tsv` to a temp folder. A row is one of:
- ok;
- rec (its own hash, recorded);
- DIFFERS (a path that must equal base doesn't);
- CHANGED (unlike `-Baseline`);
- SHORT (fewer than 16 tokens generated);
- FAIL / ERROR.

The shell's `ARC_*` / `INFERZERO_*` / `ZE_*` variables are cleared for the runs, except `ARC_DATA_DIR` and
`ZE_AFFINITY_MASK`. Until the shared layer exists, it compares fusion off / on and the paths that must equal base.
Phase 1 adds `ARC_GEMM_SHARED` on / off.

**The laptop's baseline** (125H Xe-LPG, driver 32.0.101.9175, main 204a3e6 + the harness at 43ab012; 54 runs, green
twice in a row, every hash equal between the two). **Not valid at HEAD** (the 2026-11-28 audits): since bb9458c every split row is an ERROR (only the one-GPU path printed the generation line), and the set is now 61 rows (the GDN_FUSE, INK_FUSE and moe_rg rows). Being fixed; the table is recorded again green twice on one build:

| Model | Base hash | Kinds run | Notes |
|---|---|---|---|
| Qwen3.5 4B UD-Q4_K_XL | 70906E39A9EE61BB | all | host, layer and the six fusion switches equal base; slots OK |
| Qwen3.5 9B UD-Q4_K_XL | 5E65E3BD605C2168 | all | the same |
| Qwen3.8 27B GSQ IQ3_S (MTP) | 5C199FD26C0DEF20 | all | the same. `ARC_H8=0` also gives the base hash: the 27B takes no H8 on the iGPU, so `ARC_H8_NORM` / `H8_REUSE` / `H8_SWI` check nothing there; `GEMV_GU`, `GDN_FUSE2` and `BATCH` still do |
| Qwen3.6 35B-A3B UD-IQ4_XS | EF37993715CC7D39 | no host / option (MoE); fusion: GDN_FUSE=0 and the experts without moe_rg (since 70f7ce7) | layer equals base; slots OK; both fusion rows equal base |
| Inkling 2 Mini UD-IQ4_XS | 8C9B0CF29F635EFC | base, short, fusion (INK_FUSE=0, the experts without moe_rg: per tile, its own hash DE5E8415DE35CD96) | it refuses the splits and MTP and batches no slots |

A baseline holds only for its build and driver. After a driver update, record a new one before comparing.

## D. Routes

One row per model x feature x setup: **works** (its proof), **refused** (the refusal's file:line), or **unknown**.
Filled from the Developer's check of 2026-11-27 against main 01617e2.

| Model | Feature / setup | State | Where / proof |
|---|---|---|---|

## E. Speed baseline

Same-release pairs only: the release before a migration step and the step itself, on the same card. Old issue
numbers are context. Models: the 4B, the 27B IQ3_S, the 35B-A3B, Flash Next, Gemma 4 12B and E4B, Inkling 2 Mini.
Lengths: pp512, pp4096, tg128 after 512; pp16384 for the 35B and Inkling. Three A B B A runs per model and length.
**How a step is judged** (changed 2026-11-28, after the laptop baseline showed heat dominating the small models):
- **On pairs, not against the table.** Before / after in one block on the same card: first a 2-3 minute heat soak with
  the same model, then before / after / after / before back to back. pp takes runz's 3 passes per run; tg takes 3 runs
  a side.
- **The step is judged on the pairwise difference,** which cancels the slow thermal drift. It's blocked when its median
  difference is below zero by more than max(2 %, 2 x the spread of the pairs' differences), on two such blocks.
- **The table below is the reference level:** what "normal" is on a card, so a large drop shows even without a pair.
- **Generation at depth, not after pp512 only** (the user, 2026-11-28). Each model's tg128 after prompts of 4, 512,
  4,096, 16,384, 32,768, 65,536, 131,072 tokens and its limit (the `--ctx auto` context on that card), with the per-step
  times so the slope over depth shows. The prompt speed at each length too (pp at the same lengths). A step is judged
  at every depth it reaches, with the pairs rule above.
**The laptop, Xe-LPG** (the Developer, 2026-11-28; Core Ultra 5 125H, driver 32.0.101.9175, main 089a927). Three
rounds, the model order reversed each round, 20 s pauses. pp: 9 samples (3 passes x 3 rounds); tg: 3. tok/s, median
(spread):

| Model | pp512 | pp4096 | tg128 | Bar with the old rule |
|---|---|---|---|---|
| Qwen3.5 4B UD-Q4_K_XL | 832.2 (20.9 %) | 786.3 (23.8 %) | 19.7 (5.8 %) | thermal, use pairs |
| Qwen3.5 9B UD-Q4_K_XL | 267.3 (13.3 %) | 411.3 (3.3 %) | 11.7 (9.4 %) | thermal, use pairs |
| Qwen3.8 27B GSQ IQ3_S | 103.1 (4.8 %) | 119.9 (3.1 %) | 5.0 (9.6 %) | 6-19 % |
| Qwen3.6 35B-A3B UD-IQ4_XS | 523.3 (2.3 %) | 779.7 (3.4 %) | 21.3 (1.5 %) | 3-7 % |
| Inkling 2 Mini UD-IQ4_XS | 68.2 (1.0 %) | 45.7 (2.0 %) | 8.8 (1.1 %) | 2-4 % |

The 4B is bimodal: run first on a cool laptop it gave 968-985 (pp512) and 892-941 (pp4096); every later run gave
811-833 / 753-800. The 9B's pp512 jumped the same way (~298 against ~265). The old rule's bar would be 42-48 % for
the 4B, which nothing would fail. Hence the pairs. Other cards' rows come from the kit (J).

## F. Memory plan baseline

`--ctx auto`'s choice and its memory line per model. On the laptop at `ARC_GPU_MEM_GB` = 6 / 8 / 12 / 16; on discrete
cards from the kit. **Proof:** the dry count against the real allocation after load, on the same run. Any gap is an
uncounted buffer the layer must add (first suspects: the JIT decode areas, H8's areas, the resident W8 partial share).

**The laptop, no cap** (the Developer, 2026-11-28; 125H with 20.42 GiB of shared memory, driver 32.0.101.9175, main
6fb85d7 + fix/plan-memory). The real allocation is zrt's own sum of every `zeMemAllocDevice`, printed with
`ARC_GEMM_PLAN=1`'s summary ("gemm-plan memory"): serverz after its warm-up, runz at its end. GiB throughout.

| Model | `--ctx auto` plan (weights + buffers + context) | Context chosen | serverz loaded | After the warm-up: counted / allocated | runz (ctx 1,040) at its end: counted / allocated |
|---|---|---|---|---|---|
| Qwen3.5 4B | 6.61 + 0.09 + 4.13 = 10.83 | 262,144 | 10.84 | 10.849 / 10.849 | 6.771 / 6.771 |
| Qwen3.5 9B | 12.54 + 0.14 + 4.13 = 16.81 | 262,144 | 16.80 | 16.810 / 16.810 | 5.616 / 5.616 |
| Qwen3.8 27B GSQ IQ3_S | 10.94 + 1.69 + 5.53 = 18.16 | 163,840 | 18.16 | 18.159 / 18.159 | 11.728 / 11.728 |
| Qwen3.6 35B-A3B | 14.80 + 0.72 + 2.76 = 18.28 | 262,144 | 18.19 | 18.189 / 18.190 | 16.867 / 16.868 |
| Inkling 2 Mini | 13.39 + 0.29 + 6.16 = 19.84 | 188,416 | 19.14 | 19.138 / 19.139 | 15.264 / 15.265 |

- **No uncounted buffer on any of the five:** counted equals allocated within 0.001 GiB, after the warm-up's prompt too.
  H8's areas, the staging and the buffers made at the first prompt are all in `gpu_bytes`.
- **What that proves, and what it doesn't** (the 2026-11-28 audits): `buf()` adds to `gpu_bytes` and allocates through
  `zeMemAllocDevice`, which adds to `device_bytes`, so equality shows that no device buffer bypasses `buf()`. It doesn't see the
  driver's own memory (kernel binaries, spill scratch, command lists), host or shared USM (pinned host weights, staging),
  or the allocator's rounding. On an iGPU these share the same physical memory. Comparing against the driver's own
  usage report too is for after Phase 0.
- **The plan against the load:** equal on the 4B, 9B and 27B. Two models plan high, which is safe but costs context:
  the 35B by 0.09 GiB, Inkling by 0.70 GiB. So Inkling's `--ctx auto` picks less context than fits.
- **The 27B's plan line says "left -0.59":** its parts (kept free 1.17 + snapshots 0.29 + encoders 1.39 + weights
  10.94 + buffers 1.69 + context 5.53 = 21.01) exceed the 20.42 GiB device, yet the load fits (18.16 real). The
  printed line and the fit check count the encoders or the kept-free room differently. To look at after Phase 0.
- **Still to do:** the discrete cards, from the kit.

**The laptop under a cap** (`ARC_GPU_MEM_GB`, serverz `--ctx auto`, main 089a927): context chosen, GiB loaded, the W8
form. Counted equals allocated within 0.001 GiB in all 18 runs that loaded.

| Model | 6 GB | 8 GB | 12 GB | 16 GB |
|---|---|---|---|---|
| Qwen3.5 4B | 20,480; 4.35; W8 just in time | 147,456; 9.04; resident | 229,376; 10.33; resident | 262,144; 10.84; resident |
| Qwen3.5 9B | 8,192; 3.97; just in time | **refused** | 204,800; 10.00; just in time | 262,144; 10.91; just in time |
| Qwen3.8 27B GSQ IQ3_S | refused (4.2 GiB + 1.4 for the encoders) | 8,192; 5.79; just in time | **61,440**; 9.80; just in time | **20,480**; 13.70; just in time |
| Qwen3.6 35B-A3B | 32,768; 3.76; resident | 69,632; 5.76 | 131,072; 9.76 | 131,072; 13.76 |
| Inkling 2 Mini | 32,768; 4.71; resident | 32,768; 6.71 | 65,536; 10.72 | 98,304; 14.71 |

Two findings for after Phase 0 (not changed):
- **The 9B loads at 6 GB but is refused at 8 GB.** At 8 the fit takes the heavier form (6.5 GiB without context) plus
  1.4 GiB kept for the encoders, and refuses instead of falling back to the lighter form that loads at 6 (W8 just in
  time, 3.97 GiB at 8K context). A refusal should never appear at a larger cap than one that loads.
- **The 27B gets less context with more memory:** 61,440 at 12 GB, 20,480 at 16 GB. At 12, `plan_tight` puts ~4.25
  GiB of the weights in host memory, so 2.28 GiB are left for context. At 16 all weights go on the GPU and 1.07 GiB is
  left. That's by design (`plan_tight` only when the model nearly fills the share), but it's a policy choice for the
  user: context or GPU-resident weights at that size.

## G. Launches and start-up

Launches per generated token (`ARC_LAUNCH_STATS=1`) and per 512-row chunk (`--prof`), the warm-up's time, and the
total start, after an update (a cold kernel cache). Before a cold-cache run with `ARC_DATA_DIR`, check the real hang
marker by hand: that variable moves the marker too. The per-model budget file `tests/launch_budget.txt` is new.

| Model | Card | Launches a token | Launches a 512 chunk | Warm-up (cold) | Start (cold) |
|---|---|---|---|---|---|

## H. Switches that steer GEMM / GEMV / expert choices

349 distinct `getenv("ARC_...")` names (2026-11-27). Each is listed with the function that reads it and the member it
sets. It's classified by three existing lists: `scripts/options.tsv` (user-facing), `gpu_sweep`'s settings (tuned by
contributors), and xe2-tuning.md's defaults per kind. A switch in none of the three is test-only or forgotten: a
candidate to retire, each checked. This is the seed of the switch registry (plan section 6).

### The code side (the Developer, 2026-11-28, read at main ebd3bb5)

Every `getenv("ARC_...")` site in `zero/`, `tools/` and `src/`: 475 sites of 350 switches. Of them, **154 steer a GEMM,
GEMV or expert choice**: every switch read in the H8, XMX wide-tile, MoE-XMX, host-CPU, CPU-expert and NPU files and in
`engine_moe.h`, and by name the GEMM / GEMV / W8 / XMX / MoE / stacking / two-pass / f32-GEMV switches. Left out: the
expert cache's memory switches, the DeltaNet and attention kernels, the splits' sums, the profiling labels and the
failsafes. The columns: **Read in**, the first sites (file:line); **Sets**, the functions that read it; **Default where**,
the first site's code, where the default usually is; **Class**, from the three lists.

- **By class:** 8 user-facing (`options.tsv`), 54 tuned by contributors (`gpu_sweep`), 42 a GPU kind's default
  (xe2-tuning.md), **50 in none of the three**.
- **The 50 in none:** 40 are named in a doc or the CHANGELOG (A/B and test switches with a measurement behind them).
  **10 are named nowhere outside the code**, the first candidates to retire or to list: `ARC_140T_MIN`, `ARC_DBG_MOE`,
  `ARC_G4_RG_GU`, `ARC_G4_RTILE`, `ARC_H8_E2E`, `ARC_MOE_ASCT`, `ARC_MOE_G`, `ARC_MOE_RG_MAX`, `ARC_MOE_RG_NY`,
  `ARC_RG_IMUL`. Not all are idle: `ARC_MOE_RG_MAX` sets the chunk size where decode + GEMM takes over from
  `moe_rg` (1,280), and `ARC_H8_E2E=0` skips a start check. Those two belong in a list, not in the bin.
- **Data:** `H-sites.tsv` (every site, with its function and the three lists) and `H-switches.tsv` / `H-unlisted.tsv`
  in the inventory's data folder; `scripts\gen-switch-list.ps1`'s `src/arc_switches.h` holds every name any program reads
  (serverz marks an unknown one at start).

| Switch | Read in | Sets | Steers | Default where | Class |
|---|---|---|---|---|---|
| `ARC_MOE_BIG_CHUNK` | tools/server_catalog.h:971 | moe_big_chunk_env | expert path | `static const int v = getenv("ARC_MOE_BIG_CHUNK") ? atoi(getenv("ARC_MOE_BIG_CHUNK")) : -1;` | user |
| `ARC_MOE_DPASW` | engine_setup.h:594 | moe_dpasw_on | expert path | `static const bool on = getenv("ARC_MOE_DPASW") && atoi(getenv("ARC_MOE_DPASW")) != 0;` | sweep |
| `ARC_MOE_G` | engine_alloc.h:156 | alloc_buffers | expert path | `static const int g_env = getenv("ARC_MOE_G") ? atoi(getenv("ARC_MOE_G")) : 64;` | **none: to check** |
| `ARC_MOE_GEMM_PAD` | engine_alloc.h:168, engine_moe.h:545 | alloc_buffers, moe_gemm_pad_on | expert path | `} else if (moe_gemm_pad_on() && getenv("ARC_MOE_GEMM_PAD") && !opt.dry) {  // (asked for by name: say why not)` | sweep |
| `ARC_MOE_GRF` | xe2/engine_xe2.h:224 | xe2_moe_grf | expert path | `static const char* e = getenv("ARC_MOE_GRF");  // (read once)` | sweep |
| `ARC_MOE_RG_XMX` | engine.h:523, xe2/engine_moe_xw.h:25 | init, moe_rgx_on | expert path | `const bool rg = getenv("ARC_MOE_RG_XMX") && atoi(getenv("ARC_MOE_RG_XMX")) != 0 && !rgx_emu();` | sweep |
| `ARC_MOE_X8` | engine_setup.h:585 | moe_x8_env | expert path | `static const bool on = getenv("ARC_MOE_X8") && atoi(getenv("ARC_MOE_X8")) != 0;` | sweep |
| `ARC_DBG_MOE` | engine.h:1874, engine_moe.h:507 | layer_moe, run_layer | expert path (per tile / moe_rg / decode + GEMM) | `static const int dbg_mix = getenv("ARC_DBG_MOE") ? atoi(getenv("ARC_DBG_MOE")) : -1;` | **none: to check** |
| `ARC_DUMP_ROUTES` | engine_moe.h:151 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static FILE* routes = getenv("ARC_DUMP_ROUTES") ? fopen(getenv("ARC_DUMP_ROUTES"), "ab") : nullptr;` | **none: to check** |
| `ARC_G4_RG_GU` | engine_moe.h:200 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const bool g4_rg_gu = !getenv("ARC_G4_RG_GU") \|\| atoi(getenv("ARC_G4_RG_GU")) != 0;` | **none: to check** |
| `ARC_G4_RTILE` | engine_moe.h:28 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const bool g4_rt = !getenv("ARC_G4_RTILE") \|\| atoi(getenv("ARC_G4_RTILE")) != 0;` | **none: to check** |
| `ARC_MOE_ASCT` | engine_moe.h:253 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const bool asct_on = !getenv("ARC_MOE_ASCT") \|\| atoi(getenv("ARC_MOE_ASCT")) != 0;` | **none: to check** |
| `ARC_MOE_CG` | engine_moe.h:657 | moe_gemm | expert path (per tile / moe_rg / decode + GEMM) | `static const int mcg_e = getenv("ARC_MOE_CG") ? std::clamp(atoi(getenv("ARC_MOE_CG")), 1, 4) : 0;` | sweep |
| `ARC_MOE_GEMM_MIN` | engine_moe.h:182 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const int gemm_min = getenv("ARC_MOE_GEMM_MIN") ? atoi(getenv("ARC_MOE_GEMM_MIN")) : 1152;` | per kind |
| `ARC_MOE_GR_DN` | engine_moe.h:176 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `gr_dn_e = getenv("ARC_MOE_GR_DN") ? std::clamp(atoi(getenv("ARC_MOE_GR_DN")), 1, 8) : 0;` | sweep |
| `ARC_MOE_GR_DN_S` | engine_moe.h:292 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const int gr_dn_s = getenv("ARC_MOE_GR_DN_S") ? std::clamp(atoi(getenv("ARC_MOE_GR_DN_S")), 1, 8) : 4;` | per kind |
| `ARC_MOE_GR_GU` | engine_moe.h:175 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const int gr_gu_e = getenv("ARC_MOE_GR_GU") ? std::clamp(atoi(getenv("ARC_MOE_GR_GU")), 1, 8) : 0,` | sweep |
| `ARC_MOE_RG_MAX` | engine_moe.h:189 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `rg_max = getenv("ARC_MOE_RG_MAX") ? atoi(getenv("ARC_MOE_RG_MAX")) : 1280;` | **none: to check** |
| `ARC_MOE_RG_MIN` | engine_moe.h:81, engine_moe.h:188 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const int rg_min0 = getenv("ARC_MOE_RG_MIN") ? atoi(getenv("ARC_MOE_RG_MIN")) : 320;` | per kind |
| `ARC_MOE_RG_NY` | engine_moe.h:576 | moe_rg | expert path (per tile / moe_rg / decode + GEMM) | `static const int ny_env = getenv("ARC_MOE_RG_NY") ? atoi(getenv("ARC_MOE_RG_NY")) : 0;` | **none: to check** |
| `ARC_MOE_TM` | engine_moe.h:78 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const int tm_std = getenv("ARC_MOE_TM") ? std::max(1, atoi(getenv("ARC_MOE_TM"))) : 16;  // (standar...` | sweep |
| `ARC_MOE_TM_SMALL` | engine_moe.h:88, engine_moe.h:89 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `if (!getenv("ARC_MOE_TM_SMALL")) return -1;` | per kind |
| `ARC_MOE_WY` | engine_moe.h:659 | moe_gemm | expert path (per tile / moe_rg / decode + GEMM) | `static const int mwy_e = getenv("ARC_MOE_WY") ? std::max(1, atoi(getenv("ARC_MOE_WY"))) : 1;  // (read once)` | sweep |
| `ARC_NVFP4_FOLD` | engine_moe.h:227 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const bool nv_fold = getenv("ARC_NVFP4_FOLD") && atoi(getenv("ARC_NVFP4_FOLD")) != 0;` | **none: to check** |
| `ARC_RG_IMUL` | engine_moe.h:579 | moe_rg | expert path (per tile / moe_rg / decode + GEMM) | `static const std::string imul = getenv("ARC_RG_IMUL") && atoi(getenv("ARC_RG_IMUL")) != 0 ? " -DRG_IMUL" : "";` | **none: to check** |
| `ARC_TP_RG_GU` | engine_moe.h:203 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const bool tp_rg_gu = !getenv("ARC_TP_RG_GU") \|\| atoi(getenv("ARC_TP_RG_GU")) != 0;` | per kind |
| `ARC_XLIST_SYNC` | engine_moe.h:380 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const bool list_sync = getenv("ARC_XLIST_SYNC") && atoi(getenv("ARC_XLIST_SYNC")) != 0;` | per kind |
| `ARC_XSPLIT` | engine_moe.h:312 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const bool xsplit = !getenv("ARC_XSPLIT") \|\| atoi(getenv("ARC_XSPLIT")) != 0;` | **none: to check** |
| `ARC_XSPLIT_TAB` | engine_moe.h:436 | layer_moe | expert path (per tile / moe_rg / decode + GEMM) | `static const int xs_tab = getenv("ARC_XSPLIT_TAB") ? atoi(getenv("ARC_XSPLIT_TAB")) : 1;` | per kind |
| `ARC_XSPLIT_TP` | engine_moe.h:524 | xsplit_tp | expert path (per tile / moe_rg / decode + GEMM) | `static const bool on = getenv("ARC_XSPLIT_TP") && atoi(getenv("ARC_XSPLIT_TP")) != 0;` | **none: to check** |
| `ARC_XCPU` | xe2/engine_xcpu.h:51 | xcpu_on | experts on the CPU | `static const int e = getenv("ARC_XCPU") ? (atoi(getenv("ARC_XCPU")) != 0 ? 1 : 0) : -1;` | user |
| `ARC_XCPU_Q2` | xe2/engine_xcpu.h:63 | xcpu_q2 | experts on the CPU | `static const bool on = getenv("ARC_XCPU_Q2") && atoi(getenv("ARC_XCPU_Q2")) != 0;` | user |
| `ARC_XCPU_THREADS` | xe2/engine_xcpu.h:67 | xcpu_threads | experts on the CPU | `static const int t = getenv("ARC_XCPU_THREADS") ? std::max(1, atoi(getenv("ARC_XCPU_THREADS")))` | **none: to check** |
| `ARC_MOE_RG_XEMU` | xe2/engine_moe_xw.h:21, xe2/engine_moe_xw.h:72 | rgx_defs, rgx_emu | experts on XMX | `static const bool e = getenv("ARC_MOE_RG_XEMU") && atoi(getenv("ARC_MOE_RG_XEMU")) != 0;` | per kind |
| `ARC_FN_AB32` | engine_load.h:603 | fn_ab32 | f32 GEMV forms | `static const bool env = getenv("ARC_FN_AB32") && atoi(getenv("ARC_FN_AB32")) != 0;` | **none: to check** |
| `ARC_FN_RT32` | engine_stream.h:547 | load_moe | f32 GEMV forms | `static const bool rt32 = getenv("ARC_FN_RT32") && atoi(getenv("ARC_FN_RT32")) != 0;` | **none: to check** |
| `ARC_INK_F32SG` | engine_inkling.h:699 | ink_mm | f32 GEMV forms | `static const bool sg = !(getenv("ARC_INK_F32SG") && atoi(getenv("ARC_INK_F32SG")) == 0);` | **none: to check** |
| `ARC_SELF_CHECK` | engine_setup.h:617, engine_setup.h:680 | gemm_self_check | GEMM start check | `if (getenv("ARC_SELF_CHECK") && atoi(getenv("ARC_SELF_CHECK")) == 0) return;` | **none: to check** |
| `ARC_GEMV_DB` | xe2/engine_xe2.h:274 | xe2_gemv_pf | GEMV | `static const int dbe = getenv("ARC_GEMV_DB") ? (atoi(getenv("ARC_GEMV_DB")) != 0 ? 1 : 0) : -1;` | sweep |
| `ARC_GEMV_DEEP` | xe2/engine_xe2.h:268 | xe2_gemv_pf | GEMV | `static const int deep = getenv("ARC_GEMV_DEEP") ? atoi(getenv("ARC_GEMV_DEEP")) : 0;` | sweep |
| `ARC_GEMV_GU` | engine.h:1518 | host_gemv_note | GEMV | `static const bool on = getenv("ARC_GEMV_GU") && atoi(getenv("ARC_GEMV_GU")) != 0;` | user |
| `ARC_GEMV_MT_GRF` | xe2/engine_xe2.h:395 | xe2_gemv_mt_grf | GEMV | `static const std::string e = getenv("ARC_GEMV_MT_GRF") ? getenv("ARC_GEMV_MT_GRF") : "";` | sweep |
| `ARC_GEMV_PF` | xe2/engine_xe2.h:275, xe2/engine_xe2.h:277 | xe2_gemv_pf | GEMV | `static const bool probe = getenv("ARC_GV_PROBE") != nullptr, pf_set = getenv("ARC_GEMV_PF") != nullptr;` | sweep |
| `ARC_GEMV_ROWS` | engine.h:1492, engine.h:1557 | gemv_splitk, gemv1_rows | GEMV | `static const int force_rows = getenv("ARC_GEMV_ROWS") ? atoi(getenv("ARC_GEMV_ROWS")) : 0;` | **none: to check** |
| `ARC_GEMV_SMALL_WG` | xe2/engine_xe2.h:350 | xe2_gemv_wg | GEMV | `static const bool on = getenv("ARC_GEMV_SMALL_WG") && atoi(getenv("ARC_GEMV_SMALL_WG")) != 0;` | sweep |
| `ARC_GEMV_SPLITK` | engine.h:1556, engine.h:1578 | gemv_splitk | GEMV | `static const bool on = getenv("ARC_GEMV_SPLITK") && atoi(getenv("ARC_GEMV_SPLITK")) != 0;` | per kind |
| `ARC_GEMV_SPLITK_FORCE` | engine.h:1573 | gemv_splitk | GEMV | `static const int force = getenv("ARC_GEMV_SPLITK_FORCE") ? atoi(getenv("ARC_GEMV_SPLITK_FORCE")) : 0;` | per kind |
| `ARC_GEMV_WAVE` | xe2/engine_xe2.h:303, xe2/engine_xe2.h:311 | xe2_gemv_rows, xe2_gemv_threads | GEMV | `static const int e = getenv("ARC_GEMV_WAVE") ? (atoi(getenv("ARC_GEMV_WAVE")) != 0 ? 1 : 0) : -1;` | sweep |
| `ARC_GEMV_WAVE_MT` | xe2/engine_xe2.h:329 | xe2_gemv_mt_rows | GEMV | `static const bool off = getenv("ARC_GEMV_WAVE_MT") && atoi(getenv("ARC_GEMV_WAVE_MT")) == 0;` | per kind |
| `ARC_GV_IQ3S` | engine.h:1451 | gemv1_rows | GEMV | `static const bool gv_iq3s = getenv("ARC_GV_IQ3S") && atoi(getenv("ARC_GV_IQ3S")) != 0;` | sweep |
| `ARC_GV_K4` | engine.h:1460 | gemv1_rows | GEMV | `static const int gv_k4 = getenv("ARC_GV_K4") ? atoi(getenv("ARC_GV_K4")) : -1;` | **none: to check** |
| `ARC_GV_PROBE` | engine.h:1613, xe2/engine_xe2.h:271, xe2/engine_xe2.h:275 | matmul1, xe2_gemv_pf | GEMV | `static const int gv_probe = getenv("ARC_GV_PROBE") ? atoi(getenv("ARC_GV_PROBE")) : 0;` | per kind |
| `ARC_GV_SGNPOP` | xe2/engine_xe2.h:287 | xe2_gv_sgn | GEMV | `static const std::string s = getenv("ARC_GV_SGNPOP") && atoi(getenv("ARC_GV_SGNPOP")) != 0 ? " -DGV_SGNPOP"...` | per kind |
| `ARC_GV_SMALL` | engine.h:1473 | gemv1_rows | GEMV | `static const int gv_small_env = getenv("ARC_GV_SMALL") ? atoi(getenv("ARC_GV_SMALL")) : 0;` | **none: to check** |
| `ARC_GV_SMALL_ANY` | engine.h:1474 | gemv1_rows | GEMV | `static const bool gv_small_any = getenv("ARC_GV_SMALL_ANY") && atoi(getenv("ARC_GV_SMALL_ANY")) != 0;` | **none: to check** |
| `ARC_GV4` | engine.h:1343 | gemv_v_rows | GEMV | `static const bool gv4 = !getenv("ARC_GV4") \|\| atoi(getenv("ARC_GV4")) != 0;` | **none: to check** |
| `ARC_GVROWS` | engine.h:1330, engine.h:1335, engine.h:1488 (+1) | gemv_v_rows, gemv1_rows, xe2_gemv_rows | GEMV | `const int v = getenv("ARC_GVROWS") ? atoi(getenv("ARC_GVROWS")) : 0;` | sweep |
| `ARC_HOST_GEMV_MB` | engine.h:1501 | host_gemv_mb | GEMV | `static const double env = getenv("ARC_HOST_GEMV_MB") ? std::max(0.0, atof(getenv("ARC_HOST_GEMV_MB"))) : -1.0;` | **none: to check** |
| `ARC_H8` | engine_h8.h:60, xe2/engine_xe2.h:61, xe2/engine_xe2.h:70 | h8_auto_wanted, h8_dp4a_wanted, h8_env | H8 prompt GEMM | `static const bool on = getenv("ARC_H8") && atoi(getenv("ARC_H8")) != 0;` | user |
| `ARC_H8_D` | engine_h8.h:159, engine_h8.h:204, engine_h8.h:336 (+1) | h8_norm_env, h8_out_kernel, h8r_decode, h8r_kernel | H8 prompt GEMM | `static const bool d = getenv("ARC_H8_D") && atoi(getenv("ARC_H8_D")) != 0;` | per kind |
| `ARC_H8_DEC_PRIV` | engine_h8.h:207 | h8r_decode | H8 prompt GEMM | `static const bool priv = getenv("ARC_H8_DEC_PRIV") && atoi(getenv("ARC_H8_DEC_PRIV")) != 0;` | per kind |
| `ARC_H8_DEC_R` | engine_h8.h:210 | h8r_decode | H8 prompt GEMM | `const int e = getenv("ARC_H8_DEC_R") ? atoi(getenv("ARC_H8_DEC_R")) : 4;` | per kind |
| `ARC_H8_E2E` | engine_h8.h:394 | h8_e2e_check | H8 prompt GEMM | `static const bool off = getenv("ARC_H8_E2E") && atoi(getenv("ARC_H8_E2E")) == 0;` | **none: to check** |
| `ARC_H8_FUSE` | engine_h8.h:194, engine_h8.h:198 | h8_dec_ok | H8 prompt GEMM | `static const bool on = !getenv("ARC_H8_FUSE") \|\| atoi(getenv("ARC_H8_FUSE")) != 0;` | sweep |
| `ARC_H8_GRF` | engine_h8.h:154 | h8_grf | H8 prompt GEMM | `static const int env = getenv("ARC_H8_GRF") ? atoi(getenv("ARC_H8_GRF")) : 0;` | sweep |
| `ARC_H8_HB` | engine_h8.h:110 | h8_block | H8 prompt GEMM | `const int v = getenv("ARC_H8_HB") ? atoi(getenv("ARC_H8_HB")) : 1024;` | per kind |
| `ARC_H8_INT` | engine_h8.h:76 | h8_float | H8 prompt GEMM | `static const bool i8 = getenv("ARC_H8_INT") && atoi(getenv("ARC_H8_INT")) != 0;` | per kind |
| `ARC_H8_MIN` | engine_h8.h:287 | h8_igpu_rows_ok | H8 prompt GEMM | `static const int min_rows = getenv("ARC_H8_MIN") ? std::max(0, atoi(getenv("ARC_H8_MIN"))) : 256;` | user |
| `ARC_H8_NORM` | engine_h8.h:320 | h8_norm_env | H8 prompt GEMM | `static const char* e = getenv("ARC_H8_NORM");` | sweep |
| `ARC_H8_OUT` | engine_h8.h:359 | h8_out_env | H8 prompt GEMM | `static const int e = getenv("ARC_H8_OUT") ? atoi(getenv("ARC_H8_OUT")) : -1;` | per kind |
| `ARC_H8_Q2` | engine_h8.h:46 | h8_q2_var | H8 prompt GEMM | `static const bool on = getenv("ARC_H8_Q2") && atoi(getenv("ARC_H8_Q2")) != 0;` | user |
| `ARC_H8_REUSE` | engine_h8.h:243 | h8_reuse_env | H8 prompt GEMM | `static const char* e = getenv("ARC_H8_REUSE");` | sweep |
| `ARC_H8_SB` | engine_h8.h:64 | h8_row | H8 prompt GEMM | `static const bool sb = getenv("ARC_H8_SB") && atoi(getenv("ARC_H8_SB")) != 0;` | per kind |
| `ARC_H8_SWI` | engine_h8.h:291 | h8_swi_ok | H8 prompt GEMM | `static const bool on = !getenv("ARC_H8_SWI") \|\| atoi(getenv("ARC_H8_SWI")) != 0;` | sweep |
| `ARC_HOST_CPU` | xe2/engine_hostcpu.h:54 | host_cpu_on | host-memory GEMV on the CPU | `static const int e = getenv("ARC_HOST_CPU") ? (atoi(getenv("ARC_HOST_CPU")) != 0 ? 1 : 0) : -1;` | per kind |
| `ARC_HOST_CPU_NATIVE` | xe2/engine_hostcpu.h:36 | host_native_on | host-memory GEMV on the CPU | `static const bool e = getenv("ARC_HOST_CPU_NATIVE") && atoi(getenv("ARC_HOST_CPU_NATIVE")) != 0;` | per kind |
| `ARC_HOST_CPU_THREADS` | xe2/engine_hostcpu.h:239 | host_cpu_gemv | host-memory GEMV on the CPU | `static const int nthr = getenv("ARC_HOST_CPU_THREADS") ? std::max(0, atoi(getenv("ARC_HOST_CPU_THREADS")))` | per kind |
| `ARC_INK_FUSE` | engine_inkling.h:727, engine_inkling.h:776, engine_inkling.h:803 (+2) | ink_control, ink_layer, ink_rmsnorm_q, ink_take | Inkling's GEMM forms | `static const bool unfused = getenv("ARC_INK_FUSE") && atoi(getenv("ARC_INK_FUSE")) == 0;` | **none: to check** |
| `ARC_INK_RESIDENT` | engine_inkling.h:196 | load_hparams_inkling | Inkling's GEMM forms | `static const bool resident = getenv("ARC_INK_RESIDENT") && atoi(getenv("ARC_INK_RESIDENT")) != 0;` | **none: to check** |
| `ARC_NPU` | tools/server.cpp:1541, tools/server_speech.h:79, npu/engine_npu.h:18 | npu_level, server_main | NPU share of the FFN GEMMs | `if ((so.eo.npu > 0 \|\| (so.eo.npu < 0 && getenv("ARC_NPU") && atoi(getenv("ARC_NPU")) != 0)) && ze_load()) z...` | **none: to check** |
| `ARC_NPU_DOWN` | npu/engine_npu.h:127 | npu_setup | NPU share of the FFN GEMMs | `const int nd = cols(E, getenv("ARC_NPU_DOWN") ? atof(getenv("ARC_NPU_DOWN")) : hp.gemma ? gdef : 0.275);` | **none: to check** |
| `ARC_NPU_MIN` | npu/engine_npu.h:114 | npu_setup | NPU share of the FFN GEMMs | `if (getenv("ARC_NPU_MIN")) npu_min_ = std::max(16, atoi(getenv("ARC_NPU_MIN")));` | **none: to check** |
| `ARC_NPU_QKV` | npu/engine_npu.h:128 | npu_setup | NPU share of the FFN GEMMs | `const int nq = cols(Cq, getenv("ARC_NPU_QKV") ? atof(getenv("ARC_NPU_QKV")) : 0.29);` | **none: to check** |
| `ARC_NPU_SHARE` | npu/engine_npu.h:113, npu/engine_npu.h:118 | npu_setup | NPU share of the FFN GEMMs | `const double share = getenv("ARC_NPU_SHARE") ? atof(getenv("ARC_NPU_SHARE")) : 1.0 / 3.0;` | per kind |
| `ARC_GEMM_CG` | xe2/engine_xe2.h:76, xe2/engine_xe2.h:232, xe2/engine_xe2.h:238 | h8_cg_for, xe2_cg_for | prompt GEMM | `static const bool cg_env = getenv("ARC_GEMM_CG") != nullptr;` | sweep |
| `ARC_GEMM_DEFS` | engine_setup.h:513 | gemm_extra | prompt GEMM | `static const std::string e = getenv("ARC_GEMM_DEFS") ? std::string(" ") + getenv("ARC_GEMM_DEFS") : std::st...` | **none: to check** |
| `ARC_GEMM_GRF` | xe2/engine_xe2.h:113 | xe2_gemm_grf | prompt GEMM | `static const std::string e = getenv("ARC_GEMM_GRF") ? getenv("ARC_GEMM_GRF") : "";` | sweep |
| `ARC_GEMM_PLAN` | gemm_plan.h:18 | plan_on | prompt GEMM | `static const bool on = getenv("ARC_GEMM_PLAN") && atoi(getenv("ARC_GEMM_PLAN")) != 0;` | **none: to check** |
| `ARC_GEMM_RG` | engine.h:1742, engine_h8.h:622, xe2/engine_xe2.h:886 (+2) | h8_matmul1, matmul1, x16_act, xw_gu_matmul, xw_rg | prompt GEMM | `static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;` | sweep |
| `ARC_GEMM_SPLIT` | engine.h:1752, engine_h8.h:568, xe2/engine_xe2.h:888 | h8_splits, matmul1, x16_act | prompt GEMM | `static const int sk_sb = getenv("ARC_GEMM_SPLIT") ? std::clamp(atoi(getenv("ARC_GEMM_SPLIT")), 0, 4096) : 1...` | sweep |
| `ARC_GEMM_TM` | engine_setup.h:518 | gemm_tm | prompt GEMM | `const int t = getenv("ARC_GEMM_TM") ? atoi(getenv("ARC_GEMM_TM")) : 16;` | sweep |
| `ARC_GEMM_WGY` | engine.h:1743, engine_h8.h:623, xe2/engine_xe2.h:887 (+2) | h8_matmul1, matmul1, x16_act, xw_gu_matmul, xw_tile_for | prompt GEMM | `static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;` | sweep |
| `ARC_GEMM_WPF` | xe2/engine_xe2.h:419 | xe2_wpf | prompt GEMM | `static const int n = getenv("ARC_GEMM_WPF") ? atoi(getenv("ARC_GEMM_WPF")) : 0;` | sweep |
| `ARC_X16` | xe2/engine_xe2.h:827 | x16_wanted | prompt GEMM | `return tr \|\| (getenv("ARC_X16") && atoi(getenv("ARC_X16")) != 0 && cl_.xmx && (!zrt::integrated() \|\| cl_.sg...` | sweep |
| `ARC_X16_CG` | xe2/engine_xe2.h:809 | x16_cg_for | prompt GEMM | `const int c = getenv("ARC_X16_CG") ? atoi(getenv("ARC_X16_CG")) : 1;` | per kind |
| `ARC_X16_GRF` | xe2/engine_xe2.h:817 | x16_kernel | prompt GEMM | `static const bool g128 = getenv("ARC_X16_GRF") && std::string(getenv("ARC_X16_GRF")) == "128";` | **none: to check** |
| `ARC_X16_TM` | xe2/engine_xe2.h:791 | x16_tm | prompt GEMM | `const int t = getenv("ARC_X16_TM") ? atoi(getenv("ARC_X16_TM")) : 32;` | per kind |
| `ARC_X16_TRY` | xe2/engine_xe2.h:824, xe2/engine_xe2.h:1001 | x16_check, x16_wanted | prompt GEMM | `const bool tr = getenv("ARC_X16_TRY") && atoi(getenv("ARC_X16_TRY")) != 0;` | per kind |
| `ARC_INK_STACK` | engine_inkling.h:240 | ink_stack_load | stacked matrices | `static const bool off = getenv("ARC_INK_STACK") && atoi(getenv("ARC_INK_STACK")) == 0;  // (A/B: the matric...` | **none: to check** |
| `ARC_NO_GU_FLT` | engine_load.h:803 | load_weights | stacked matrices | `} else if ((hp.qwen3 \|\| (float_type(f_.get(p + "ffn_gate.weight").type) && !getenv("ARC_NO_GU_FLT"))) &&` | **none: to check** |
| `ARC_NO_QKV_FLT` | engine_stream.h:785 | load_attn | stacked matrices | `const bool stack = (hp.qwen3 \|\| (float_type(f_.get(p + "attn_q.weight").type) && !getenv("ARC_NO_QKV_FLT"))...` | **none: to check** |
| `ARC_NO_QZ_FLT` | engine_load.h:399 | load_gdn_in | stacked matrices | `if (float_type(pq->type) && float_type(pz->type) && !getenv("ARC_NO_QZ_FLT")) {` | **none: to check** |
| `ARC_NO_QZAB16` | engine_load.h:424 | load_gdn_in | stacked matrices | `L.qz_ab = !(hp.hc && fn_ab32() && !tp_on()) && type == GGML_Q8_0 && nab % 16 == 0 && !getenv("ARC_NO_QZAB16");` | **none: to check** |
| `ARC_FN_Q2` | engine.h:1393 | q2_init | two-pass GEMMs | `q2mm_ = (q2_ && !g4_one) \|\| (hp.hc && getenv("ARC_FN_Q2") && atoi(getenv("ARC_FN_Q2")) > 0) \|\| (hp.qwen3 &&...` | **none: to check** |
| `ARC_G4_ONE` | engine.h:1389 | q2_init | two-pass GEMMs | `(getenv("ARC_G4_ONE") && atoi(getenv("ARC_G4_ONE")) != 0));` | **none: to check** |
| `ARC_G4_Q2MM` | engine.h:1388 | q2_init | two-pass GEMMs | `const bool g4_one = hp.gemma && ((hp.n_ple > 0 && !(getenv("ARC_G4_Q2MM") && atoi(getenv("ARC_G4_Q2MM")) !=...` | **none: to check** |
| `ARC_Q3_Q2` | engine.h:1392 | q2_init | two-pass GEMMs | `const bool q3_one = getenv("ARC_Q3_Q2") && atoi(getenv("ARC_Q3_Q2")) == 0;` | **none: to check** |
| `ARC_IQ3S_TILE` | engine_load.h:105 | iq3t_env | weight form (W8 resident / just in time, tiles) | `static const bool on = getenv("ARC_IQ3S_TILE") && atoi(getenv("ARC_IQ3S_TILE")) != 0;` | sweep |
| `ARC_W8_MINS` | engine_load.h:592 | w8_mins_on | weight form (W8 resident / just in time, tiles) | `static const int env = getenv("ARC_W8_MINS") ? atoi(getenv("ARC_W8_MINS")) : -1;` | **none: to check** |
| `ARC_W8_PARTIAL_GB` | engine_load.h:207, engine_load.h:212, engine_load.h:240 | w8_part_auto, w8_part_auto_env, w8_part_take | weight form (W8 resident / just in time, tiles) | `static const bool v = !getenv("ARC_W8_PARTIAL_GB") \|\| std::string(getenv("ARC_W8_PARTIAL_GB")) == "auto";` | sweep |
| `ARC_W8_RESIDENT` | engine_setup.h:394 | plan_weights | weight form (W8 resident / just in time, tiles) | `static const bool keep = getenv("ARC_W8_RESIDENT") != nullptr;  // (A/B)` | per kind |
| `ARC_140T` | xe2/engine_xe2.h:80, xe2/engine_xe2.h:95 | p140_off_resident, p140_wanted | XMX / GPU kind | `const char* e = getenv("ARC_140T");` | sweep |
| `ARC_140T_MIN` | xe2/engine_xe2.h:27 | p140_min | XMX / GPU kind | `static const int e = getenv("ARC_140T_MIN") ? atoi(getenv("ARC_140T_MIN")) : 0;` | **none: to check** |
| `ARC_140T_RES` | engine_setup.h:437 | plan_weights | XMX / GPU kind | `static const bool res = getenv("ARC_140T_RES") && atoi(getenv("ARC_140T_RES")) != 0;` | user |
| `ARC_NO_SG8` | clrt.h:229, clrt.h:1450 | batch_default_for, init | XMX / GPU kind | `if (getenv("ARC_NO_SG8") && atoi(getenv("ARC_NO_SG8")) != 0) sg8 = false;` | per kind |
| `ARC_NO_XMX` | clrt.h:1470 | init | XMX / GPU kind | `if (getenv("ARC_NO_XMX") && atoi(getenv("ARC_NO_XMX")) != 0) xmx = false;` | sweep |
| `ARC_XMX_FORCE` | engine.h:474, engine_setup.h:855 | init, xmx_check | XMX / GPU kind | `const bool xmx_force = getenv("ARC_XMX_FORCE") && atoi(getenv("ARC_XMX_FORCE")) != 0;` | sweep |
| `ARC_XMX_SPEED` | engine_setup.h:854 | xmx_check | XMX / GPU kind | `const bool spd_env = getenv("ARC_XMX_SPEED") && atoi(getenv("ARC_XMX_SPEED")) != 0;` | per kind |
| `ARC_XMX_TRY` | engine.h:476, engine.h:480, engine_setup.h:526 (+5) | gemm_kernel, init, moe_gemm_kernel, p140_wanted, xe2_grf128, xmx_check, xmx_defs | XMX / GPU kind | `if (!opt.dry && cl_.xmx && sg8_igpu && !xmx_force && !p140_ && !getenv("ARC_XMX_TRY")) {` | per kind |
| `ARC_140T_BIG` | xe2/engine_xw.h:493 | p140_big_env | XMX wide-tile GEMM (gemm_xw) | `static const bool on = getenv("ARC_140T_BIG") && atoi(getenv("ARC_140T_BIG")) != 0;` | per kind |
| `ARC_GEMM_XW` | xe2/engine_xw.h:68 | xw_wanted | XMX wide-tile GEMM (gemm_xw) | `const char* e = getenv("ARC_GEMM_XW");` | per kind |
| `ARC_GEMM_XW_2D` | xe2/engine_xw.h:225, xe2/engine_xw.h:1153 | xw_2d_level, xw_init | XMX wide-tile GEMM (gemm_xw) | `static const char* e = getenv("ARC_GEMM_XW_2D");` | sweep |
| `ARC_GEMM_XW_BG` | xe2/engine_xw.h:660 | xw_bg_env | XMX wide-tile GEMM (gemm_xw) | `static const bool on = getenv("ARC_GEMM_XW_BG") && atoi(getenv("ARC_GEMM_XW_BG")) != 0;` | per kind |
| `ARC_GEMM_XW_BIG` | xe2/engine_xw.h:505 | xw_big_on | XMX wide-tile GEMM (gemm_xw) | `static const int env = getenv("ARC_GEMM_XW_BIG") ? atoi(getenv("ARC_GEMM_XW_BIG")) : -1;` | sweep |
| `ARC_GEMM_XW_C_RND` | xe2/engine_xw.h:1086 | xw_init | XMX wide-tile GEMM (gemm_xw) | `static const char* rnd_env = getenv("ARC_GEMM_XW_C_RND");` | sweep |
| `ARC_GEMM_XW_DB` | xe2/engine_xw.h:254 | xw_db_env | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_DB") ? std::max(0, std::min(2, atoi(getenv("ARC_GEMM_XW_DB")))) : -1;` | sweep |
| `ARC_GEMM_XW_DIAG` | xe2/engine_xw.h:1358 | xw_check | XMX wide-tile GEMM (gemm_xw) | `static const bool diag = getenv("ARC_GEMM_XW_DIAG") && atoi(getenv("ARC_GEMM_XW_DIAG")) != 0;` | **none: to check** |
| `ARC_GEMM_XW_DPASW` | xe2/engine_xw.h:630 | xw_dpasw_env | XMX wide-tile GEMM (gemm_xw) | `static const char* e = getenv("ARC_GEMM_XW_DPASW");` | per kind |
| `ARC_GEMM_XW_EPI2` | xe2/engine_xw.h:283 | xw_epi2 | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_EPI2") ? (atoi(getenv("ARC_GEMM_XW_EPI2")) != 0 ? 1 : 0) : -1;` | sweep |
| `ARC_GEMM_XW_GRF` | xe2/engine_xw.h:47 | xw_grf256 | XMX wide-tile GEMM (gemm_xw) | `static const char* e = getenv("ARC_GEMM_XW_GRF");` | **none: to check** |
| `ARC_GEMM_XW_GU` | xe2/engine_xw.h:240, xe2/engine_xw.h:859 | xw_gu_env, xw_gu_matmul | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_GU") ? (atoi(getenv("ARC_GEMM_XW_GU")) != 0 ? 1 : 0) : -1;` | sweep |
| `ARC_GEMM_XW_K128` | xe2/engine_xw.h:201 | xw_k128 | XMX wide-tile GEMM (gemm_xw) | `static const bool v = getenv("ARC_GEMM_XW_K128") && atoi(getenv("ARC_GEMM_XW_K128")) != 0;` | **none: to check** |
| `ARC_GEMM_XW_LANES` | xe2/engine_xw.h:34 | xw_lanes | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_LANES") ? atoi(getenv("ARC_GEMM_XW_LANES")) : 0;` | **none: to check** |
| `ARC_GEMM_XW_LOCK` | xe2/engine_xw.h:195 | xw_lock | XMX wide-tile GEMM (gemm_xw) | `static const bool v = getenv("ARC_GEMM_XW_LOCK") && atoi(getenv("ARC_GEMM_XW_LOCK")) != 0;` | sweep |
| `ARC_GEMM_XW_LSC` | xe2/engine_xw.h:656, xe2/engine_xw.h:1145 | xw_init, xw_lsc_opt | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_LSC") ? atoi(getenv("ARC_GEMM_XW_LSC")) : 0;` | per kind |
| `ARC_GEMM_XW_NC` | xe2/engine_xw.h:53, xe2/engine_xw.h:506, xe2/engine_xw.h:996 | xw_big_on, xw_nc, xw_shape_on | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_NC") ? atoi(getenv("ARC_GEMM_XW_NC")) : 0;` | sweep |
| `ARC_GEMM_XW_NC_E` | xe2/engine_xw.h:971 | xw_nc_e | XMX wide-tile GEMM (gemm_xw) | `const int e = getenv("ARC_GEMM_XW_NC_E") ? atoi(getenv("ARC_GEMM_XW_NC_E")) : 0;` | sweep |
| `ARC_GEMM_XW_PACK` | xe2/engine_xw.h:146 | xw_pack_env | XMX wide-tile GEMM (gemm_xw) | `const char* e = getenv("ARC_GEMM_XW_PACK");` | per kind |
| `ARC_GEMM_XW_PF` | xe2/engine_xw.h:79, xe2/engine_xw.h:94, xe2/engine_xw.h:97 (+1) | xw_pf, xw_pf_eff, xw_pf_max_def, xw_pfd | XMX wide-tile GEMM (gemm_xw) | `const int e = getenv("ARC_GEMM_XW_PF") ? atoi(getenv("ARC_GEMM_XW_PF")) : 1;` | sweep |
| `ARC_GEMM_XW_PFD` | xe2/engine_xw.h:86, xe2/engine_xw.h:94, xe2/engine_xw.h:97 | xw_pf_max_def, xw_pfd | XMX wide-tile GEMM (gemm_xw) | `const int e = getenv("ARC_GEMM_XW_PFD") ? atoi(getenv("ARC_GEMM_XW_PFD")) : 1;` | sweep |
| `ARC_GEMM_XW_PFL` | xe2/engine_xw.h:128 | xw_pfl_form | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_PFL") ? std::clamp(atoi(getenv("ARC_GEMM_XW_PFL")), 0, 2) : -1;` | sweep |
| `ARC_GEMM_XW_PTR` | xe2/engine_xw.h:483 | xw_ptr | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_PTR") ? (atoi(getenv("ARC_GEMM_XW_PTR")) != 0 ? 1 : 0) : -1;` | sweep |
| `ARC_GEMM_XW_SHAPE` | xe2/engine_xw.h:995 | xw_shape_on | XMX wide-tile GEMM (gemm_xw) | `static const int env = getenv("ARC_GEMM_XW_SHAPE") ? atoi(getenv("ARC_GEMM_XW_SHAPE")) : -1;` | per kind |
| `ARC_GEMM_XW_SLM` | xe2/engine_xw.h:271, xe2/engine_xw.h:507 | xw_big_on, xw_slm_wgy | XMX wide-tile GEMM (gemm_xw) | `const int v = getenv("ARC_GEMM_XW_SLM") ? atoi(getenv("ARC_GEMM_XW_SLM")) : 0;` | **none: to check** |
| `ARC_GEMM_XW_SLM_CPS` | xe2/engine_xw.h:327, xe2/engine_xw.h:337, xe2/engine_xw.h:341 | xw_big_make | XMX wide-tile GEMM (gemm_xw) | `(!getenv("ARC_GEMM_XW_SLM_CPS") \|\| atoi(getenv("ARC_GEMM_XW_SLM_CPS")) == 1)) {` | sweep |
| `ARC_GEMM_XW_SLM_TILE` | xe2/engine_xw.h:314, xe2/engine_xw.h:326, xe2/engine_xw.h:337 | xw_big_make | XMX wide-tile GEMM (gemm_xw) | `if (const char* t = getenv("ARC_GEMM_XW_SLM_TILE")) {  // (TMxCOLS, each 16 / 32 / 64: xw_peak --all's forms)` | sweep |
| `ARC_GEMM_XW_SLM_WG` | xe2/engine_xw.h:321, xe2/engine_xw.h:386 | xw_big_gu_make, xw_big_make | XMX wide-tile GEMM (gemm_xw) | `if (const char* w = getenv("ARC_GEMM_XW_SLM_WG")) {` | sweep |
| `ARC_GEMM_XW_SLM_WG_GU` | xe2/engine_xw.h:380 | xw_big_gu_make | XMX wide-tile GEMM (gemm_xw) | `if (const char* w = getenv("ARC_GEMM_XW_SLM_WG_GU")) {` | per kind |
| `ARC_GEMM_XW_SLM_WG_ROLES` | xe2/engine_xw.h:407 | xw_role_cfg | XMX wide-tile GEMM (gemm_xw) | `const char* e = getenv("ARC_GEMM_XW_SLM_WG_ROLES");` | per kind |
| `ARC_GEMM_XW_SPLIT` | xe2/engine_xw.h:479 | xw_split_env | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_SPLIT") ? atoi(getenv("ARC_GEMM_XW_SPLIT")) : -1;` | per kind |
| `ARC_GEMM_XW_TM` | xe2/engine_xw.h:40, xe2/engine_xw.h:506, xe2/engine_xw.h:996 | xw_big_on, xw_shape_on, xw_tm | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_TM") ? atoi(getenv("ARC_GEMM_XW_TM")) : 0;` | sweep |
| `ARC_GEMM_XW_TRY` | xe2/engine_xw.h:62 | xw_try | XMX wide-tile GEMM (gemm_xw) | `static const bool t = getenv("ARC_GEMM_XW_TRY") && atoi(getenv("ARC_GEMM_XW_TRY")) != 0;` | **none: to check** |
| `ARC_GEMM_XW_WGL` | xe2/engine_xw.h:608, xe2/engine_xw.h:679 | xw_wgl_b, xw_wgl_env | XMX wide-tile GEMM (gemm_xw) | `static const int e = getenv("ARC_GEMM_XW_WGL") ? atoi(getenv("ARC_GEMM_XW_WGL")) : -1;` | per kind |
| `ARC_GEMM_XW_WGX` | xe2/engine_xw.h:178, xe2/engine_xw.h:179, xe2/engine_xw.h:186 (+3) | xw_big_on, xw_shape_on, xw_wgx | XMX wide-tile GEMM (gemm_xw) | `if (!getenv("ARC_GEMM_XW_WGX")) return 0;` | sweep |

## I. Dead paths

A form is a deletion candidate only if all three hold:
- it never appears in any GPU kind's plan log (A);
- it isn't one of B's guards (a fallback, a low-memory form, a failure path);
- no `options.tsv` row or `gpu_sweep` setting reaches it.

It's deleted **during** migration, when its route reaches "old path deleted", with C's hashes equal. Not before.

### The code side (the Developer, 2026-11-28, read at main ebd3bb5)

The laptop's plan logs (A, Xe-LPG) saw **60 of the 200 kernels** in `zero/kernels/` (and `rd_block` / `rd_strip`, the
read kernels whose source is in `engine.h`). Every one of the other 140 is
named by host code (none is left without a caller), so none is dead by the first test alone. The plan logs of the
other GPU kinds come from the kit (J), and the models the laptop does not run (Gemma 4, Flash Next, DeepSeek-V4.1,
the vision and audio encoders, Whisper) from their own runs. So most unseen kernels are **not decided yet**:
- **A model's own kernels, unseen because the laptop doesn't run that model or path** (decided on its runs):
  `ops_inkling.cl` 23 (Inkling 2's DSpark drafter, its loop and BF16 forms), `ops_flashnext.cl` 20, `ops_vit.cl` 19,
  `ops_ds41.cl` 19, `whisper.cl` 15, `ops_attn_g.cl` 10 (Gemma 4's attention), `ops_dflash.cl` 5, `asr.cl` 4, and
  `moe_gateup` / `moe_down` (the 2-bit GSQ experts), `moe_gelu_mul` (Gemma 4's experts), `mtp_concat_hc` (Flash
  Next's MTP block), `wh_ln_quant` / `wh_gelu_quant` (Whisper).
- **A GPU kind's kernels, unseen on Xe-LPG** (decided by the kit): `gemm_x16.cl` 2, `gemm_x8.cl`, `gemm_xw.cl`,
  `gemm_xw8.cl`, `attn_fx` (XMX flash attention), `gdn_chunk_x` / `gdn_chunk_x8`, `h8_unwgl` (`ARC_GEMM_XW_WGL`),
  `gated_norm_h8r` / `attn_gate_h8r` (`ARC_H8_OUT`, Alchemist's default), `h8_act` / `h8_weights` (`ARC_H8_SB=1`,
  per-256 H8; both switches are a kind's default in xe2-tuning.md).
- **Reached by a switch or setup in a list, not by the laptop's defaults:** `attn_fa` (`--attn-f32`, and QSA's float
  path), `qsa_fa_combine` (QSA), `attn_ps` (long prompts where the per-GPU table `qwen_ps` turns it on),
  `attn_split` (decode rows of 128+ lanes: drafters, wide head groups), `gdn_chunk` (`ARC_GDN_CHUNK`, gpu_sweep),
  `tp_rs_add` (`ARC_TP_OVERLAP`'s sums).

**Candidates now**, forms that only a switch in no list reaches (H), or that no model reaches:

| Form | Never chosen on | Not a guard because | Not reachable because | Deleted in |
|---|---|---|---|---|
| `moe_rg`'s `-DRG_IMUL` variant (`decode_moe.cl`) | Xe-LPG (A); no kind's default sets it | another form of the same product, an experiment's | only `ARC_RG_IMUL=1`, named in no list, doc or CHANGELOG (H) | the experts' step, when `moe_rg` moves |
| `ARC_DBG_MOE`'s dump (`engine_moe.h`, `moe_dbg_gpu.bin`) | any kind (debug output) | writes a file, computes nothing | only `ARC_DBG_MOE=layer`, in no list (H) | when `layer_moe` moves |
| `gemv -DQT=40`, the dense NVFP4 one-token GEMV | every kind: no model has a dense NVFP4 matrix (Inkling 2's NVFP4 tensors are its experts) | no type falls back to it | only a model with a dense NVFP4 matrix: none today. It spills 512-2,912 bytes on Xe-LPG (`gemm_check`) | the GEMV step, unless a model needs it first (then made spill-free) |

The other unlisted switches (H) reach no form of their own: `ARC_MOE_ASCT=0` takes the tile kernel's form without
the transposed scales, which the one-token tiles use anyway; `ARC_G4_RTILE=0` and `ARC_G4_RG_GU=0` take Gemma 4's
other existing forms; `ARC_140T_MIN`, `ARC_MOE_G` and `ARC_MOE_RG_NY` move a threshold or a launch shape;
`ARC_H8_E2E=0` skips a start check. **Data:** `I-kernels-all.txt`, `I-seen-mtl.txt`, `I-unref.tsv` (each unseen
kernel, its file and its host references) in the inventory's data folder.


## J. The contributor kit

`bench/contrib_kit.ps1` / `.sh`, through `gpu_guard`, 10-20 minutes, one file to attach:
- the device, driver and release;
- A's plan log;
- E's pp512 / pp4096 / tg128 x 2;
- **generation at depth** (2026-11-28): pp and tg128 after 4,096, 32,768 and 131,072 tokens and at the model's limit on that card, once each (a `-Depth` option, ~10-30 more minutes a model; off with `-Quick`);
- G's launch counts;
- F's `--ctx auto` memory line;

for whichever of E's models the contributor has. The first ask goes to skipzx-lab (A770, B60), hibagus (B70),
manoreken2 (4 x B60) and LukaszBrzyszkiewicz (B70, A380) once it's merged.

## K. Fusions: what is fused today, and where fused GEMMs go

Today's fused forms are where most of the per-GPU tuning lives, so the shared layer must carry each one as a fusion
rule, not as a separate kernel next to it. One row per fused form in use:
- **fused GEMMs:**
  - gate + up in one launch: `gemv_gu` / `-DGU2`, `ARC_GEMM_XW_GU`;
  - stacked projections (qkv + z, Inkling's q / rel / gate / k / v, the MoE router + shared gate / up);
  - SwiGLU in the up GEMM's epilogue (`ARC_H8_SWI`);
  - the MoE experts' gate / up with situ;
  - `ink_ctrl_fused`;
- **fused prologues:** the norm and H8 rotation in one kernel (`rmsnorm_h8r`, `ARC_H8_NORM`); one H8 activation
  shared by several GEMMs (`ARC_H8_REUSE`); `attn_gate_h8r`; the quantization kernels that write H8's float copy
  (`-DH8F`);
- **fused epilogues and other steps:** the DeltaNet conv / gates fused into the recurrence (`-DCONV_UPD`,
  `-DFUSE_GATES`); the outputs rotated for H8 by the kernels that make them (`ARC_H8_OUT`); the KV store and norms in
  attention's preparation; the multi-GPU sums' fused forms (`tp.cl`).

**H8R's end-to-end error grows with a row's mean over K** (one scale a row after the rotation; `gemm_check`,
2026-11-28). The rotation puts a rotation block's mean into one coefficient, of size |mean| x sqrt(block), and the
row's single scale must cover it, so every other value of the row gets coarser:
- **Synthetic rows:** with centred blocks, H8R's end-to-end error is a median 3.4x the one-token GEMV's (max 8.8x). With
  block means far from 0 (random mins: `gemm_check`'s `end_to_end_offsets` column, run every time) it rises to 12-38x
  for the types with a min (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K, Q2_0). The W8 GEMM stays at 1.0x either way.
- **Real rows** (`gemm_check --weights`: the 4B, the 27B, the 35B-A3B, every matrix, experts included) sit at the
  centred level. The largest rotation block's |mean| x sqrt(block) / deviation has a 99th percentile of 2.2-4.3 per type,
  against 2.9-3.7 centred and 24-38 with random mins.
- **The outliers:** the DeltaNet output projections (`ssm_out`) have the largest maxima, 6.8-10.9 (layer 0 the largest
  in the 35B and the 27B), about 3x the centred level. That's where H8R loses most on today's models; a catalog rule for
  H8 should know it.

### The code side (the Developer, 2026-11-28, read at main 6fb85d7)

Each fused form in use: the unfused chain it replaces, its switch and default, where it's chosen, whether the laptop's plan logs saw it (A), and its proof from the golden set's fusion rows (C: each must equal the model's base hash). "Taken on the laptop" is the Xe-LPG kind only. Other kinds follow the code's defaults and stay unproven until the kit. Spill status per family comes from the kit's start lines (A).

#### Fused GEMMs

| fused form | replaces | switch (default) | chosen at | taken on the laptop | proof |
|---|---|---|---|---|---|
| gate+up GEMV (`gemv` -DGU2) | the gate and the up GEMVs, two launches | `ARC_GEMV_GU=1` (off: opt-in) | engine.h:1517 (gemv_gu's rules: B 1) | no (opt-in) | golden fusion ARC_GEMV_GU=1 = base on the 4B, 9B, 27B |
| gate+up+SwiGLU GEMM (`gemm_xw` -DXW_GU, role gemm_gateup) | the gate GEMM, the up GEMM and swiglu_quant | `ARC_GEMM_XW_GU` (the default per kind: discrete XMX) | engine_qwen.h:499; xe2/engine_xw.h:231 | no (no XMX) | unproven on the laptop: xw_check at start (B 3), the kit on XMX cards |
| qkv+z stacked GEMV / GEMM (`gemm_qkvz`) | the qkv and the z products, two launches | none: taken when attn_qkv and attn_gate share a type and the shapes fit | engine_load.h:405-455 (the stack), engine_qwen.h:22 | yes: 27B, 35B (the 4B / 9B UD files mix types: two products) | no off switch, so no A/B hash; exact by construction (the same rows and kernel over one buffer) |
| router+shared gate/up GEMV (`gemm_router_sh`, Layer::rgu) | the router, the shared expert's gate and up: three GEMVs | none: taken when the shared gate / up are Q8_0, E % 256, not w8_resident on a dense model | engine_stream.h:560-595, engine_moe.h:65 | yes: 35B | no off switch, no A/B hash; exact by construction |
| both int8 passes in one weight read (`gemv` -DQ2F) | the two-pass GEMV run twice | none: two-pass models (q2mm_: Gemma, Qwen3) at T <= 4 | engine.h:1415-1416, 1454, 1623 | no (no two-pass model on the laptop) | unproven here: needs a Gemma 4 / Qwen3 file; run_tests 4c covers Gemma's KL |
| expert decode + GEMM in registers (`moe_rg`) | decode_w8 of the experts, then the expert GEMM | none to turn off; ARC_MOE_RG_MIN / _MAX move its row range (320 / 1280) | engine_moe.h:188-197 (rg_ok), 212-219, 584 | yes: 35B, Inkling | golden 35B `ARC_MOE_RG_MIN=1000000 ARC_MOE_GEMM_MIN=1` = base: every expert chunk then takes decode_w8_sl + gemm `-DMOE_TILES` (plan log), so moe_rg equals decode + GEMM bit for bit. Inkling's same row takes the per-tile kernels (its NVFP4 / IQ4_NL down layers), a different path with its own hash (DE5E8415DE35CD96): not counted as proof, its KL to measure with the Inkling work |
| Inkling's controller gain + forget + apply (`ink_ctrl_fused`) | three steps, separate launches | `ARC_INK_FUSE=0` turns it off (on by default) | engine_inkling.h:727, 818-824 | yes: Inkling | golden `ARC_INK_FUSE=0` = base on Inkling |

#### Fused prologues (norm, quantization, rotation)

| fused form | replaces | switch (default) | chosen at | taken on the laptop | proof |
|---|---|---|---|---|---|
| norm+rot (`rmsnorm_h8r`) | rmsnorm_quant (with its f32 copy) then h8r_act's rotation | `ARC_H8_NORM` (p140_, discrete non-Xe-HPC, or h8_dp4a_ with hb <= 512) | engine_h8.h:319-332 | yes: 4B (9B no: hb 1024) | golden ARC_H8_NORM=0 = base on the 4B (taken); the 9B and 27B rows pass without taking it |
| one rotation shared by the GEMMs of one input | a rotation before each GEMM | `ARC_H8_REUSE` (on) | engine_h8.h:238-255, 467-470 | yes: 4B, 9B | golden ARC_H8_REUSE=0 = base on the 4B, 9B |
| swiglu+rot (the down GEMM's rotation reads gate / up) | swiglu_quant, then the rotation of its output | `ARC_H8_SWI` (on where H8R has its f32 copy) | engine_h8.h:290-293, engine_qwen.h:511 | yes: 4B, 9B | golden ARC_H8_SWI=0 = base on the 4B, 9B |
| attn gate+rot / gated norm+rot (`attn_gate_h8r`, `gated_norm_h8r`) | the gate / gated norm quant, then the rotation for o / ssm_out | `ARC_H8_OUT` (default: discrete 8-lane, Alchemist) | engine_h8.h:347-375, engine_qwen.h:105, 470 | no | unproven on the laptop (not taken by default; ARC_H8_OUT=1 not in golden) |
| quant+f32 copy (`-DH8F`) | a separate f32 copy of the rows H8R rotates from | none: on whenever H8R takes its f32 source (h8_float: not ARC_H8_INT, not hc / gemma) | engine.h:562, engine_h8.h:77, 87, 475 | yes: 4B, 9B | inside every H8 run: golden base of the 4B / 9B; no A/B of its own |
| norm+quant (`rmsnorm_quant`), swiglu+quant, gated norm+quant, attn gate+quant | the norm / activation, then a quantization kernel | none: the only form there is | engine.h:1768-1780; engine_qwen.h:114, 480, 520 | yes: every model | no unfused form exists to compare; they are the reference chain |
| add+norm(+quant) (`add_norm_quant`, `add_rmsnorm`) | the residual add, then the norm (and its quantization) | none | engine.h:1911-1992, engine_gemma.h:169 | no (Gemma 4's sandwich norms; the drafter's add_rmsnorm) | unproven on the laptop (no Gemma file) |
| q/k norm + RoPE + KV store (`attn_prep`, `attn_prep_g`) | the per-head norms, RoPE, the cache writes (and the int8 copies for attn_q8) | none | engine.h:1959; engine_gemma.h:39 | yes: the Qwen models | no unfused form; the reference |

#### Fused epilogues and other steps

| fused form | replaces | switch (default) | chosen at | taken on the laptop | proof |
|---|---|---|---|---|---|
| conv + state update (`gdn_conv` -DCONV_UPD) | the short conv, then the conv history update | `ARC_GDN_FUSE` (on) | engine_load.h:36; engine_qwen.h:39, 51-52 (one token, not in verification passes) | yes: the Qwen models | golden `ARC_GDN_FUSE=0` = base on the 4B, 9B, 27B, 35B |
| recurrence + gates (`gdn_recur` / `gdn_chunk` -DFUSE_GATES) | gdn_gates, then the recurrence | `ARC_GDN_FUSE` (on; not in verification passes: `!spec_`) | engine_load.h:56; engine_qwen.h:39, 77, 89 | yes: the Qwen models | golden `ARC_GDN_FUSE=0` = base on the 4B, 9B, 27B, 35B |
| two recurrence steps in one sweep (`gdn_recur` -DGDN_FUSE2) | the second pass of token t and the first of t + 1 | `ARC_GDN_FUSE2=1` (off: opt-in; not with ARC_GDN_PF) | engine_load.h:50-54 | no (opt-in) | golden ARC_GDN_FUSE2=1 = base on the 4B, 9B, 27B |
| the multi-GPU sums' fused forms (tp.cl) | a sum, then the residual add / norm | per split (choose_sums, ARC_TP_*) | zero/mgpu/engine_tp.h | yes under ARC_TP_SIM (tp2 / tp3) | golden tp2 / tp3: their own hashes, stable over two runs; not against one GPU (not bit-identical by design) |

**Gaps in the proof:**
- **Closed 2026-11-28** (d96c538): `ARC_GDN_FUSE=0` and `ARC_INK_FUSE=0` equal base, and `moe_rg` equals decode + GEMM on the 35B. Inkling's experts without moe_rg take the per-tile kernels, a different path with its own hash; its KL waits for the Inkling work (an f32 reference of the Mini has never been run).
- **Not taken on the laptop:** `XW_GU` and `H8_OUT` (XMX / Alchemist: from the kit); `Q2F` and `add_norm_quant` (they need a Gemma 4 or Qwen3 file).
- **No switch, now tested by `gemm_check`** (merged 2026-11-28, e055af2): the qkv+z and router+shared stacks equal their separate products bit for bit (18 / 18 on the 4B, 36 / 36 on the 27B, at 1, 2-8 and prompt rows), and `moe_rg` equals decode + GEMM (9 / 9 on the 35B, three layers by three routings). `-DH8F`, the norm+quant family and `attn_prep` are the reference chain itself.
- **Every fusion row now shows whether its form ran** (the golden `form` column). `ARC_GEMV_GU=1` and `ARC_GDN_FUSE2=1` are taken and equal base. Not taken, so no proof there: `H8_NORM` on the 9B and 27B, `H8_SWI` on the 27B (no H8 step). `ARC_BATCH=64` is a switch row (command-list batching), not a fusion.
- **Found by `gemm_check`:** `add_norm_quant` (Gemma 4) differs from add_rmsnorm + rmsnorm_quant in the last bits, against its CHANGELOG claim ("Output is identical to the two separate kernels"): on the after-Phase-0 list. `moe_down_q` at 8 slots differs from 16 in the last bits (GROWS 4 against 1), which no claim covers: recorded behaviour. NVFP4's one-token gemv spills 512-2,912 bytes on Xe-LPG (no model takes it).

**Where fused GEMMs go in the shared layer** (design section 3.12):
- A fused GEMM is **not a kernel of its own**. It's the shared GEMM stage plus a prologue (`Norm`, `Quant`, the H8
  rotation) and an epilogue (`Act`, `Add`, `Combine`, `Softcap`, quantization for the next GEMM), generated from the
  layer's chain.
- **Which neighbouring steps join is a fusion rule** in each GPU kind's catalog, with its measurement, not code spread
  through `engine_h8.h` and `xe2/`.
- **Every fused form must equal its unfused chain bit for bit**, checked by `gemm_check` per rule. So a fusion choice
  changes only speed, never output.
- **New kernels are only fusion kernels** (the user, 2026-11-26, #92): a model or a change adds a fusion rule, or a
  step no kernel covers, never another GEMM.
- **Grouped products** (one launch over weights side by side, the design's "way 1") and **horizontal fusion**
  (independent small steps in one launch, "way 6") are fusion rules too.

**Per model** (the user, 2026-11-27: "for each models"). For each model:
- which of the fused forms above it takes today, per GPU kind, from its plan log (A);
- its launches a token now (G);
- its floor, the launches left when every legal fusion is taken (design section 3.7);
- its planned chain. The floors and chains are from design section 3.7, and the "now" counts are to be measured in G.

| Model | Fused forms today (from A) | Launches a token now | Floor (ways 1-8; the lowest, ways 1-12: design 3.7a) | Planned chain, per layer |
|---|---|---|---|---|
| Qwen3.5 4B / 9B (dense, 3 DeltaNet : 1 attention) | Xe-LPG: 4B attn gate+quant; conv+state update; gated norm+quant; norm+quant; norm+rot; q/k norm+rope+kv store; quant+f32 copy; recurrence+gates; swiglu+quant; swiglu+rot; 9B the same but no norm+rot (its rows get a rotation block of 1,024, above the default's 512) | 4B ~426 (125H `--prof-gen`, 2026-11-26) | ~5 a layer: 4B ~160 | DeltaNet: [norm -> qkvz + ab, conv / state / gates in the epilogue] · [recurrence] · [gated norm -> ssm_out + residual] · [norm -> gate / up, SwiGLU] · [down + residual]; attention: [norm -> q / k / v, q / k norm, RoPE, KV store] · [attention] · [sigmoid gate -> o + residual] · the FFN's 2 |
| Qwen3.8 27B | Xe-LPG: no H8 forms (no H8 on the iGPU); attn gate+quant; conv+state update; gated norm+quant; norm+quant; q/k norm+rope+kv store; recurrence+gates; swiglu+quant; qkv+z | ~710 | **321** | as the 4B, 48 DeltaNet + 16 attention layers |
| Qwen3.6 35B-A3B (MoE) | Xe-LPG: as the 27B, plus decode+expert gemm (moe_rg) and router+shared gate/up | ~620 | **241** | the mixer's 3, then [norm -> router + shared gate / up] · [top-k -> experts' gate / up, the shared down grouped in] · [experts' down]; the combine in the next layer's prologue |
| Flash Next | to fill | ~1,050 | **~483** | the hc mixer's 2 global steps per block, then the mixer's or the MoE's 3; `hc_combine` in the block's last epilogue |
| DeepSeek-V4.1 | to fill | ~1,900 | **~415** | the design's 10-13 steps a layer (mix, Sinkhorn + norm -> q_a / kv grouped, attention, out_a in one launch, out_b + expand, the MoE's 3) |
| Gemma 4 dense (31B) | to fill | ~840 | **301** | as Qwen3 dense; the post-attention norm + residual + FFN norm in gate / up's prologue |
| Gemma 4 26B MoE | to fill | to measure | 6 a layer | qkv · attention · o · [dense FFN gate / up + router] · [experts' gate / up + dense down] · [experts' down] |
| Inkling 2 Mini | Xe-LPG: controller; decode+expert gemm; norm+quant; swiglu+quant (the fewest) | **1,356** | **385** | every visit: [mix + norm -> q / gate / rel / k / v / sink / kcp / vcp grouped, K / V convs in the epilogue] · [attention] · [o + post-conv + residual] · [FFN mix + norm -> router] · [top-k -> experts' gate / up, situ] · [experts' down]; later passes add the controllers' 4 |
| the encoders (vision, audio, speech) | to fill | to measure | to work out | per encoder, once their GEMMs move to the layer |

The gap between "now" and the floor is each model's fusion work. Inkling's is the largest (3.5x), which is one reason it
goes first.

Phase 0 records each of today's fused forms in this table with its proof. Phase 1 carries each one as a rule over the
same kernels, bit-identical. The new fusions (the design's ways 1-8: fewer launches toward each model's floor) come
after a model is on the layer, Inkling first.

## Exit rule for Phase 0

1. every row of A-K has a source and a proof, or says "unproven";
2. the silent refusals are fixed (Inkling on `--split layer`, Gemma 4's `--dflash`), with the five stale comments;
3. C runs green twice in a row on the same build and driver, and E's baseline is taken on that build;
4. F shows no uncounted buffer, or lists the ones the layer must count;
5. only the discrete kinds' A / E / F / G rows may stay "unproven" until the kit comes back. Phase 1 then starts with
   what the laptop can check: the host layer over unchanged kernels (the design's section 6.1);
6. the user signs off.

Phase 0's code, smallest first, each its own branch with C's hashes equal:
1. the silent refusals and comments;
2. `ARC_GEMM_PLAN=1`;
3. C's harness;
4. the kit;
5. optionally, a cache-only override and a compile counter.
