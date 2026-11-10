# 本地验证记录

日期：2026-11-10。环境：macOS arm64、Python3.11、CPU。确切依赖见`requirements-tested.txt`。

## 自动化测试

`python -m pytest -q`：21项通过。

- HH多轮解析、连续同角色合并、角色标记前空格、正文角色字样保留。
- 标签反转、completion边界、keep-end截断及padding屏蔽。
- λ=0/1端点的损失与梯度、λ=0.5组合、完全校正的固定margin期望恒等式、非法参数。
- 小型随机Qwen2模型的LoRA梯度、冻结底座、batch-size不变性。
- CLI端到端两步训练（含gradient checkpointing）、中间checkpoint、最终适配器保存和重载评估。
- 协议不匹配、输出覆盖、RR种子错配的拒绝行为。
- RR确定性复现、不同种子与预算的随机流分离、9组配对计划。

`ruff check --select E9,F`通过。

## 实际数据和tokenizer

使用真实`Qwen/Qwen2.5-1.5B-Instruct` tokenizer，未加载1.5B权重，对原项目所有6500条数据执行编码审计：

| 划分 | 样本数 | chat失败数 | raw_hh失败数 | 超过1536长度的pair |
|---|---:|---:|---:|---:|
| train | 5000 | 0 | 0 | 0 |
| validation | 500 | 0 | 0 | 0 |
| test | 1000 | 0 | 0 | 0 |

只检查格式和token长度，没有计算测试集效果或用于选择模型。

可重新运行：

```bash
python scripts/audit_tokens.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/processed/train.jsonl data/processed/validation.jsonl data/processed/test.jsonl \
  --output-json runs/token_audit.json
```

本次发现原数据有5条需要处理非交替或缩进的角色标记。合并连续同角色段落、兼容行首空格后无需删除样本。正文内缺失换行的角色字样仍作为正文保留，没有猜测重写原意。

数据下载入口在本地实际执行过，解压校验成功，并重建了与原项目一致的5000/500/1000划分。三个processed文件SHA256：

```
train      0f9daedc27899b11d543cf7c70324462c0aae5f3cdbc043b868f433197c35f0b
validation a3bf4aaa7ac84d56e9f65b6d8711bc0a4b024b93521de246a578cbee2f4b548b
test       c6983fb445721a6008fbce31e48ec4f326c500a992ed5c8648c5def159d9de6c
```

ε=2、RR种子42/43/44的真实5000条扰动数据已在本地生成；9组训练计划已生成，未执行正式训练。

## 尚未验证

- 真实1.5B模型训练效果、旧适配器结果、新方法收益。
- Ascend/CUDA设备运行、BF16数值行为和实际显存消耗。
- 生成质量、统计显著性和最终测试效果。

本次代码与CPU测试通过，不等于上述实验已经完成。
