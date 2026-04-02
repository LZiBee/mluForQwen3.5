# Qwen3.5 Phase 2 进展记录（2026-04-02）

## 背景

当前目标是在 `vLLM v0.6.2 + vllm_mlu` 分支上推进 `Qwen3.5` 适配，但当前环境还没有 MLU 板卡，因此优先完成所有不依赖硬件的工作。

Phase 2 的重点不是直接接 `Qwen3_5ForConditionalGeneration`，而是先把 `Qwen3.5` 文本骨干依赖迁到当前分支上，让错误从“配置/接口缺失”推进到更深的模型骨架和底层执行层。

## 本轮完成内容

### 1. 配置识别基座

已在本地工作分支补齐以下配置类：

- `Qwen3_5Config`
- `Qwen3_5TextConfig`
- `Qwen3_5MoeConfig`
- `Qwen3_5MoeTextConfig`
- `Qwen3NextConfig`

同时已完成 registry 注册，使当前分支能够识别：

- `qwen3_5`
- `qwen3_5_text`
- `qwen3_5_moe`
- `qwen3_5_moe_text`
- `qwen3_next`

### 2. Phase 2 审计脚本

新增审计脚本：

- `vllm-v0.6.2/tools/qwen3_5_phase2_audit.py`

用途：

- 用固定检查项判断当前分支距离 `qwen3_next + GDN + hybrid executor` 还差哪些依赖
- 每完成一轮 backport 后可重复执行，快速看到缺口变化

### 3. model executor 浅层接口

已补充的浅层接口和 helper：

- `IsHybrid`
- `MixtureOfExperts`
- `EagleModelMixin`
- `extract_layer_index`
- `sequence_parallel_chunk`

这一步的目的不是让模型直接运行，而是先消除导入期就报缺类、缺协议、缺 helper 的问题。

### 4. 文档与测试基座

已新增：

- `Qwen3.5 Phase 2` 迁移计划文档
- `Qwen3.5` 配置测试
- `Qwen3Next` 配置测试

## 验证结果

本轮在当前宿主环境执行过的检查：

```bash
python3 -m py_compile \
  vllm-v0.6.2/vllm/transformers_utils/configs/qwen3_next.py \
  vllm-v0.6.2/vllm/transformers_utils/config.py \
  vllm-v0.6.2/vllm/transformers_utils/configs/__init__.py \
  vllm-v0.6.2/vllm/model_executor/models/interfaces.py \
  vllm-v0.6.2/vllm/model_executor/models/utils.py \
  vllm-v0.6.2/tests/test_qwen3_next_config.py
```

结果：

- `py_compile` 通过

审计脚本结果从：

- `1/17 checks satisfied`

提升到：

- `7/17 checks satisfied`

说明当前工作已经把 Phase 2 从“配置识别/接口缺失”推进到了“准备补模型骨架”的阶段。

## 当前仍缺失的关键项

审计脚本当前仍报告缺失：

- `vllm/model_executor/models/qwen3_next.py`
- `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
- `SharedFusedMoE`
- `vllm/model_executor/layers/fla/ops/__init__.py`
- `vllm/model_executor/layers/mamba/abstract.py`
- `vllm/model_executor/layers/mamba/mamba_mixer2.py`
- `RMSNormGated`
- `PluggableLayer`
- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/backends/gdn_attn.py`

## 当前结论

当前阶段已经可以明确：

- `Qwen3.5` 不是单文件 backport
- 不能靠 `TransformersForCausalLM` 作为最终方案
- 正确路径应当是：
  1. 先补 `qwen3_next` config 和浅层协议
  2. 再补 `qwen3_next.py` 文本骨架
  3. 再迁 `SharedFusedMoE`
  4. 最后处理 `gdn_linear_attn` 及其 `FLA / mamba / v1 attention` 依赖簇

## 环境限制

当前宿主 Python 环境还缺少以下依赖：

- `torch`
- `transformers`
- `pytest`

因此本轮验收主要基于：

- 静态代码落地
- `py_compile`
- 审计脚本缺口收敛

而不是运行时推理验证。

## 下一步建议

下一轮优先继续推进：

1. 迁移 `vllm/model_executor/models/qwen3_next.py` 的纯 Python 骨架
2. 补 `SharedFusedMoE` 和与其直接耦合的最小依赖
3. 最后再进入 `GatedDeltaNetAttention` 与 MLU 专项适配
