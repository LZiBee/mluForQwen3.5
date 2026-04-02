# Qwen3.5 MLU Backport 调研总结与实施路线

## 1. 背景与目标

目标是在当前 `enginex-mlu370-vllm` 的基础上，让框架原生支持 `Qwen3.5`，并尽量走 vLLM 原生模型实现路线，而不是依赖 `TransformersForCausalLM` 兜底。

当前已知约束：

- 基础仓库版本为 `vLLM v0.6.2 + vllm_mlu`。
- 当前环境启动 `Qwen3.5` 时，最先报错在 `transformers.AutoConfig` 解析 `model_type=qwen3_5` 失败。
- 目标方向是补齐 `Qwen3.5` 的配置、模型注册、文本骨干和必要的 MLU 适配，最终支持 `Qwen3_5ForConditionalGeneration`。

## 2. 当前结论

### 2.1 不能从 `scripts.py` 开始

启动报错虽然堆栈里经过 `vllm/scripts.py`，但真正失败点不在 CLI，而在配置解析层：

- `vllm/transformers_utils/config.py`

这里会先调用 `AutoConfig.from_pretrained(...)`。如果 `transformers` 不认识 `qwen3_5`，流程在进入 vLLM 模型执行层之前就已经终止。

### 2.2 `Qwen3_5ForConditionalGeneration` 不是第一刀

从上游实现看，`Qwen3.5` 的真正难点不是顶层 `ConditionalGeneration` 外壳，而是文本骨干：

- `Qwen3.5` 文本层是 hybrid 结构。
- `layer_types` 中同时存在 `linear_attention` 和 `full_attention`。
- 线性注意力依赖 `GatedDeltaNetAttention` 和一套 mamba state 逻辑。
- 当前 `v0.6.2 + vllm_mlu` 分支里没有这套能力。

因此，如果只先补一个 `Qwen3_5ForConditionalGeneration` 壳子，最终仍然会卡在内部文本模型无法实例化或无法执行。

### 2.3 当前仓库其实已经具备多模态顶层配置下沉能力

虽然还没有 `Qwen3.5`，但当前工作仓库已经具备类似的结构能力：

- `vllm/config.py` 中已经有 `hf_text_config`
- `vllm/config.py` 中已经有 `with_hf_config(...)`
- `vllm/model_executor/models/llama4.py` 已经演示了如何从顶层 `ConditionalGeneration` 配置下沉到文本骨干
- `vllm/model_executor/models/qwen2_audio.py`
- `vllm/model_executor/models/idefics3.py`

这意味着：

- 顶层 `Qwen3_5ForConditionalGeneration` 的框架接法是有现成模式可复用的。
- 真正的高风险点在 `Qwen3.5` 文本骨干与 MLU 算子兼容，而不是多模态外壳。

### 2.4 现有 `Qwen3` / `Qwen3MoE` 经验可以直接复用一部分

当前工作仓库中已经有：

- `vllm/model_executor/models/qwen3.py`
- `vllm/model_executor/models/qwen3_moe.py`

其中 `Qwen3MoE` 的适配文档与实现已经验证了几件重要事情：

- MLU 上 QK Norm 需要关注 `.contiguous()` / `.reshape()`
- MLU 上 RoPE 调用和 native 签名不同，通常需要把 `q/k` 拼接为单个 3D tensor 再调用
- `FusedMoE.forward_mlu` 签名需要和主调侧对齐

这些经验后续可直接迁移到 `Qwen3.5` 的 MLU 适配中。

## 3. 当前最核心的技术判断

如果目标是“治本”，即在当前 MLU 分支中原生支持 `Qwen3.5`，建议路线是：

1. 先补 `qwen3_5` / `qwen3_5_moe` 配置体系
2. 再补 `Qwen3.5` 文本骨干
3. 然后再接 `Qwen3_5ForConditionalGeneration`
4. 最后做 MLU 适配和验证

不建议顺序：

1. 先补 `Qwen3_5ForConditionalGeneration`
2. 再回头处理文本骨干

原因是第二种顺序会产生一个能注册但不能跑的外壳，调试路径更散。

## 4. 推荐实施步骤

### Phase 1: 补配置层，让框架先“认得出” Qwen3.5

目标：先消除 `AutoConfig` 无法识别 `qwen3_5` 的问题。

建议动作：

- 提升 `transformers` 版本到支持 `qwen3_5` 的版本
- 从上游引入：
  - `vllm/transformers_utils/configs/qwen3_5.py`
  - `vllm/transformers_utils/configs/qwen3_5_moe.py`
- 修改：
  - `vllm/transformers_utils/configs/__init__.py`
  - `vllm/transformers_utils/config.py`

预期结果：

- 模型配置至少可以被成功读取
- 运行时错误从“配置不认识”推进到“模型实现缺失”或“下一级能力缺失”

### Phase 2: 只做 `Qwen3.5` 文本骨干，不先碰视觉塔

目标：先让 dense text / moe text 能在 vLLM 模型层正确实例化。

建议优先 backport：

- 上游 `vllm/model_executor/models/qwen3_5.py`
- 上游 `vllm/model_executor/models/qwen3_next.py`

需要重点关注的依赖：

- `GatedDeltaNetAttention`
- mamba state shape / dtype / copy 相关逻辑
- hybrid 层类型 `layer_types`
- `HasInnerState` / `IsHybrid` 等接口适配

这是整个项目的主风险区。

### Phase 3: 再接 `Qwen3_5ForConditionalGeneration`

目标：在文本骨干可用的前提下，补齐顶层多模态模型。

实现思路：

- 复用 `Llama4ForConditionalGeneration` 的顶层 config 处理方式
- 复用 `Qwen2AudioForConditionalGeneration`、`Idefics3ForConditionalGeneration` 的多模态 embedding merge 结构
- 参考上游 `Qwen3_5ForConditionalGeneration`

这一阶段建议重点解决：

- `vision_config + text_config` 的顶层配置接入
- 权重名前缀映射
- `language_model` / `visual` 的模块组织

### Phase 4: MLU 适配

目标：让 `Qwen3.5` 在寒武纪算子组合上真实可跑。

优先检查项：

- `GatedDeltaNetAttention` 是否已有 MLU 路径
- hybrid 模型中的 inner state 与 prefix cache / scheduler 配合
- QK Norm 在 MLU 上的张量连续性问题
- rotary 调用签名差异
- 如果涉及 MoE，再复用 `Qwen3MoE` 已验证的 `FusedMoE.forward_mlu` 修复经验

## 5. 需要重点复用的现有仓库文件

配置与基础设施：

- `vllm/transformers_utils/config.py`
- `vllm/transformers_utils/configs/__init__.py`
- `vllm/config.py`

文本模型参考：

- `vllm/model_executor/models/qwen3.py`
- `vllm/model_executor/models/qwen3_moe.py`
- `vllm/model_executor/models/llama4.py`

多模态顶层模型参考：

- `vllm/model_executor/models/qwen2_audio.py`
- `vllm/model_executor/models/idefics3.py`
- `vllm/model_executor/models/qwen2_vl.py`

MLU / MoE 参考：

- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/models/qwen3_moe.py`

## 6. 上游参考源

建议以以下上游文件为主参考进行搬运：

- `vllm/model_executor/models/qwen3_5.py`
- `vllm/model_executor/models/qwen3_next.py`
- `vllm/transformers_utils/configs/qwen3_5.py`
- `vllm/transformers_utils/configs/qwen3_5_moe.py`

如果继续做多模态：

- `vllm/model_executor/models/qwen3_vl.py`

## 7. 风险评估

### 风险最高

- `Qwen3.5` hybrid 文本骨干依赖的 GDN / mamba state 能力，在当前 `v0.6.2 + vllm_mlu` 分支中缺口最大

### 中风险

- 多模态视觉塔本身 backport 量较大
- `transformers` 升级后对现有 MLU 分支的兼容性可能引入额外回归

### 低风险

- 顶层 `ConditionalGeneration` 外壳接线
- `text_config` 下沉
- 配置注册

## 8. 建议的最小验证顺序

建议每个阶段都只验证一个目标，避免问题叠加：

1. `AutoConfig.from_pretrained("/model", trust_remote_code=True)` 能通过
2. dense text `Qwen3.5` 能完成模型构建
3. dense text `Qwen3.5` 能完成单轮文本推理
4. `Qwen3_5ForConditionalGeneration` 能完成模型构建
5. 多模态输入链路能完成前向
6. MLU 上完成真实推理验证

## 9. 当前建议的实际开工顺序

最终建议按下面顺序推进：

1. 先做 Phase 1：补 `qwen3_5` / `qwen3_5_moe` config 与注册
2. 再做 Phase 2：补 `Qwen3.5` 文本骨干
3. 然后做 Phase 3：补 `Qwen3_5ForConditionalGeneration`
4. 最后做 Phase 4：MLU hijack 与性能版本适配

一句话总结：

`Qwen3_5ForConditionalGeneration` 不是起点，配置层和 hybrid 文本骨干才是起点；多模态外壳反而是后半段工作。
