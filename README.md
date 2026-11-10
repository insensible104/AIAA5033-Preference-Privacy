# AIAA5033 Preference Privacy

Report03项目的后续代码：统一训练与评估格式、部分噪声校正、多RR种子配对实验。

## 19组正式实验入口

主方案为1组干净DPO，加ε=1/2×3个RR种子×λ=0/0.5/1，共19组。固定第3轮为主结果，按1250/2500/3750步保留checkpoint。完整命令、资源口径、统计解释及生成盲评流程见[运行说明](docs/RUN_MAIN19.md)。配置见[main19.json](configs/main19.json)，逐组清单见[main19_matrix.csv](configs/main19_matrix.csv)。

```bash
python scripts/prepare_data.py
python scripts/run_main19.py plan --suite runs/main19_v1 --dtype bfloat16
# 只打印训练命令；加--execute才会运行
python scripts/run_main19.py train --suite runs/main19_v1 --device npu
```

旧的`run_sweep.py`继续保留，用于自由探索；课程主结果使用`run_main19.py`，避免混用配置。

## 本版与旧实验的关系

- `chat`默认模式：将HH历史解析为真正的多轮消息，使用模型自己的聊天模板。相邻同角色段落合并；行首角色标记容许空格；正文内的角色字样不猜测为新消息。
- `raw_hh`兼容模式：保留原始HH字符串，用于旧适配器的格式对照。它不是重现旧版有mask问题的评估。
- 训练和评估直接调用同一个编码器、collator和completion log-prob函数；截断统一为keep-end，padding不计分，无法计分或回答边界不匹配时明确报错。
- 新训练默认关闭所有dropout，固定底座作为参考模型，仅优化LoRA。使用显式的`transformers.Trainer`损失实现，不依赖TRL内部预处理。它不是原始Report03环境的逐位复现。
- ε决定RR噪声率；λ决定训练时校正强度，两者独立。λ=0是RR-DPO，λ=1是RR-rDPO。中间值是两种损失的凸组合，不是已经验证有效的新算法，也不保证损失有下界。
- 每个(ε,RR种子)使用独立的确定性随机流，同一条件下各方法复用同一个JSONL。与Report03跨预算复用uniform的实现不同。
- 新结果应单独成表。旧报告结果不随本次代码修改而被重新验证。

## 安装

Python3.10或以上；本地测试使用Python3.11、CPU。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

精确的本地验证依赖见`requirements-tested.txt`。Ascend环境需要另行安装与CANN匹配的PyTorch及torch-npu；本仓库不替代厂商环境安装流程，NPU运行尚未验证。CUDA/NPU训练请仅暴露一个设备；不支持DDP或DeepSpeed。不要在现有Ascend环境里直接套用macOS测试依赖清单。

## 数据

GitHub不提交原始数据、权重、训练输出。原始HH-RLHF数据来源说明见`docs/DATASET_SOURCE.md`；数据版本和解压后SHA256固定在`configs/data.json`。本地工作目录已放置原项目的干净划分，远程克隆后运行：

```bash
python scripts/prepare_data.py
python scripts/prepare_rr.py \
  --train-data data/processed/train.jsonl --output-dir data/rr \
  --epsilons 1 2 --rr-seeds 42 43 44
```

`prepare_data.py`下载缺失原始数据、验证解压后校验值，再重建5000/500/1000划分。已有文件不同则停止，不覆盖。`prepare_rr.py`生成`.jsonl`与`.meta.json`，训练会核对ε、RR种子和哈希，防止配置与数据错配。

这是公开数据模拟。公开种子、翻转数量及干净标签不适用于真正的私有发布。独立RR种子不意味着可以免费发布多份私有标签；多次发布需要隐私组合分析。

## 单次训练

先设置实际模型目录或模型ID；远程模型建议通过`--revision`固定提交。所有实验使用相同底座。

```bash
MODEL=/path/to/Qwen2.5-1.5B-Instruct

python scripts/train_dpo.py \
  --model "$MODEL" --train-data data/processed/train.jsonl \
  --group dpo --seed 20261030 --epochs 3 \
  --device cpu --dtype float32 --output-dir runs/clean

python scripts/train_dpo.py \
  --model "$MODEL" --train-data data/rr/train_rr_eps_2_seed_42.jsonl \
  --group partial --epsilon 2 --rr-seed 42 --correction 0.5 \
  --seed 20261030 --epochs 3 --gradient-checkpointing \
  --device cpu --dtype float32 --output-dir runs/partial_eps2_rr42
```

CPU示例用于说明参数，1.5B正式训练建议使用加速设备。CUDA使用`CUDA_VISIBLE_DEVICES=0`及`--device cuda --dtype bfloat16`；Ascend使用`ASCEND_RT_VISIBLE_DEVICES=0`及`--device npu --dtype bfloat16`。BF16需要设备支持。

`--group rr_dpo`固定λ=0，`--group rr_rdpo`固定λ=1；这两组都需要`--epsilon`和`--rr-seed`，不接受额外λ。`--group partial`要求显式`--correction`。

每250步保存checkpoint，最多保留最近4个，最终适配器位于`adapter/`。可用`--save-steps`调整。输出目录必须为空，避免覆盖旧实验。`run_config.json`记录参数、依赖版本、数据哈希和格式版本；`token_audit.json`记录每条样本的长度、合并段落数和截断情况。训练日志记录loss和梯度范数。

## 多种子配对实验

先生成9组实验计划（ε=2，3个RR种子，λ=0/0.5/1，固定训练种子）。此命令会生成RR文件和计划，不加载模型或启动训练：

```bash
python scripts/run_sweep.py \
  --model "$MODEL" --train-data data/processed/train.jsonl \
  --data-dir data/rr --output-dir runs/plan_eps2
```

要执行，使用一个新的输出目录，并加`--execute --device cuda --dtype bfloat16`。任务顺序执行，失败立即停止。可设置`--epsilons`、`--rr-seeds`、`--training-seeds`和`--corrections`。相同RR文件和训练种子对应各个λ；干净DPO用单独命令训练，不按RR种子重复计算。

## 评估

默认读取适配器的训练配置和tokenizer，FP32/eager attention评估。格式、β、长度不允许无意中改变。中间checkpoint读取父目录配置。

```bash
python scripts/evaluate.py \
  --adapter-dir runs/partial_eps2_rr42/adapter \
  --eval-data data/processed/validation.jsonl \
  --device cpu --output-json runs/partial_eps2_rr42/validation.json
```

换机器后可用`--model`指定同一底座的新路径。务必保持相同权重。对于无新版元数据的旧适配器，显式指定四项参数：

```bash
python scripts/evaluate.py \
  --adapter-dir /path/to/old/adapter --model "$MODEL" \
  --input-format raw_hh --max-length 1536 --beta 0.1 \
  --eval-data data/processed/validation.jsonl --output-json runs/legacy_raw_eval.json
```

新版适配器如需做格式对照，使用`--allow-protocol-override`，结果会记录这一标记。输出包含每条样本ID、margin、正确性和截断统计；零margin按0.5分计。该指标衡量固定候选的隐式奖励排序，不能替代生成质量。最终测试集应在选择配置后使用。

## 已验证与待验证

本地测试覆盖解析、截断、padding、标签反转、部分校正数值及梯度、固定参考模型、多RR种子与防覆盖，以及小型随机Qwen模型的训练→checkpoint→适配器保存→重载评估。真实Qwen tokenizer的全量样本检查见`docs/VALIDATION.md`。

尚未运行真实1.5B训练、旧适配器重评或CUDA/NPU测试，也没有生成任何新模型效果结论。
