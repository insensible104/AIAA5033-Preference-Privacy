# 19组主实验运行说明

若同时做带噪模型选择，请先按[扩展协议](RUN_SELECTION_FORWARD.md)登记选择计划，并在任何测试或生成评估前冻结选择结果与干净验证参照。本页19组固定最终checkpoint的主结果定义不变。

本协议面向课程项目的主结果。先完成受控比较，再评估部分校正；不要求部分校正必胜。配置为`configs/main19.json`，展开清单为`configs/main19_matrix.csv`。

## 固定设计

- 1组干净DPO。
- ε∈{1,2}×RR种子∈{42,43,44}×λ∈{0,0.5,1}，共18组隐私训练。
- 全部训练种子固定为20261030。三个RR种子不是三个训练种子，也不是三个独立验证集。
- 模型Qwen2.5-1.5B-Instruct，数据5000/500/1000；同一RR条件的三个λ共享完全相同的JSONL。
- 3轮、batch4、累积1，单设备共3750步；cosine、5% warmup、学习率1e-4、β=0.1、LoRA rank16/alpha32，dropout关闭。
- 在1250/2500/3750步保存checkpoint，主结果固定使用最终`adapter/`；中间checkpoint只用于训练过程诊断。
- 全部偏好评估使用FP32、eager attention和与训练一致的编码。量化/分布式训练不在本协议内。

## 1 安装与准备数据

在克隆后的仓库根目录执行，先按服务器环境安装PyTorch/CANN/torch-npu，再安装本项目。不要直接把macOS的requirements-tested.txt安装到Ascend环境。

```bash
pip install -e '.[test]'
pytest -q
python scripts/prepare_data.py
```

首次准备数据会下载并验证固定版本。已有原包数据时，可把原始JSONL放进`data/raw/`再运行。所有数据及运行输出被Git忽略。

## 2 生成19组计划

用新目录保存这一轮完整实验。下面只准备配置和6份RR数据，不加载模型权重、不训练：

```bash
python scripts/run_main19.py plan \
  --suite runs/main19_v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16
```

远程模型ID会查询并固定实际提交SHA；也可加`--revision`。本地权重使用`--model /path/to/Qwen2.5-1.5B-Instruct`，运行期间保持该目录不变。`--dtype`在计划阶段固定，不在训练阶段临时更改。

计划记录模型版本、全部训练参数、数据校验值和19个任务。拒绝重复使用非空suite目录。生成计划后可以先检查：

```bash
python scripts/run_main19.py train --suite runs/main19_v1 --device npu
```

没有`--execute`只打印命令。计划中的绝对路径用于同一服务器运行；换服务器时在目标机重新生成计划，不手动混用旧状态文件。

## 3 启动训练

Ascend示例（先加载服务器已有的CANN环境）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/run_main19.py train \
  --suite runs/main19_v1 --device npu --execute
```

CUDA示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_main19.py train \
  --suite runs/main19_v1 --device cuda --execute
```

顺序执行19个独立任务。输出写入`suite/models/<name>/`，日志写入`suite/logs/`。CPU仅适合小模型验证，正式1.5B训练需要加速设备。

如果使用多个设备，可以在各进程分别传入**互不重叠**的`--jobs`名称，例如：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 python scripts/run_main19.py train \
  --suite runs/main19_v1 --device npu --jobs clean_dpo --execute

ASCEND_RT_VISIBLE_DEVICES=2 python scripts/run_main19.py train \
  --suite runs/main19_v1 --device npu \
  --jobs eps_1_rr_42_lambda_0 eps_1_rr_42_lambda_0.5 eps_1_rr_42_lambda_1 --execute
```

脚本不提供同一任务的并发锁；不要让两个进程执行相同任务。只暴露一个设备，不使用DDP。

重跑训练命令会校验完成标记和适配器哈希，然后跳过已成功任务。失败任务的非空目录不会自动覆盖，也不会自动从checkpoint续训。检查对应日志后，将该失败任务的目录另存为备份，保留其余已完成结果，再重试。不要把失败任务手动标成完成。

## 4 验证集及训练过程评估

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/run_main19.py validation \
  --suite runs/main19_v1 --device npu --execute

python scripts/summarize_main19.py \
  --suite runs/main19_v1 --split validation \
  --output-dir runs/main19_v1/summary_validation
```

共评估每组3个checkpoint和最终adapter。checkpoint-3750与adapter预期相同，但分别保留最终导出和checkpoint的证据。每次评估重新加载模型，优先保证简单和可检查，因此评估总耗时需要单独记录。

已有输出只在数据、训练协议和适配器哈希匹配时跳过。不要将旧报告评估JSON放进这里。汇总要求19份最终结果齐全，缺失就报错；不把部分完成的实验当作完整结果。

输出：

- `runs.csv`：19组准确率、margin均值/中位数/5%与95%分位数、平局率及训练成本。
- `rr_summary.csv`：每个ε/λ的三个RR种子均值与样本标准差。
- `paired_comparisons.json`：各RR种子内部，λ=0.5对0、1对0，以及0.5对1的配对差值和逐样本bootstrap区间。
- `paired_rr_summary.csv`：上述配对差值在三个RR种子上的均值与样本标准差。
- `trajectory.csv`：存在的checkpoint结果，用于画训练过程图；缺失的checkpoint不补造。
- `summary.md`：可读汇总。

RR标准差不是置信区间。bootstrap在固定训练模型对内按样本ID配对重采样，不能解释成跨RR种子或训练随机性的置信区间。三份RR实验也不能合并为1500条独立验证样本。主分析不按中间checkpoint挑赢家。

训练`train_metrics.json`记录训练器时间、包含加载的pipeline时间及加速设备峰值allocated/reserved内存。CPU内存指标为null。评估另记包含加载的时间与纯打分时间。日志与汇总均不声称覆盖下载、排队等项目总成本。

## 5 冻结后执行最终测试

检查19组最终验证结果，决定按预注册的第3轮终点报告，再执行：

```bash
python scripts/run_main19.py freeze --suite runs/main19_v1 --execute

ASCEND_RT_VISIBLE_DEVICES=0 python scripts/run_main19.py test \
  --suite runs/main19_v1 --device npu --execute

python scripts/summarize_main19.py \
  --suite runs/main19_v1 --split test \
  --output-dir runs/main19_v1/summary_test
```

freeze要求19组训练完成且最终验证身份匹配，写入配置哈希与适配器哈希。测试仅评估19个最终adapter。配置或权重变化后拒绝使用旧freeze。该功能用于记录实验决策，不是权限控制；不要通过绕过脚本反复查看测试集来调参。

## 6 生成质量评估与人工盲评

预注册使用100个固定测试prompt，抽样种子20261110，greedy decoding，最多256个新token。比较初始模型、干净DPO，以及ε=1/2下RR种子42的三个λ，共8个模型。种子42是在看结果之前选定；这部分不能代表跨RR种子的生成稳定性。

freeze之后分别生成8份结果。先以两份为例：

```bash
python scripts/generate_main19.py --suite runs/main19_v1 \
  --name initial --device npu --output-json runs/main19_v1/generation/initial.json
python scripts/generate_main19.py --suite runs/main19_v1 \
  --name clean_dpo --device npu --output-json runs/main19_v1/generation/clean_dpo.json
python scripts/generate_main19.py --suite runs/main19_v1 \
  --name eps_2_rr_42_lambda_0.5 --device npu \
  --output-json runs/main19_v1/generation/eps_2_rr_42_lambda_0.5.json
```

其他名称见配置的`generation.names`。先逐个生成，再构造比较。用干净DPO作为统一参照，7个候选各100项，共700项人工判断；初始模型也是候选之一。下面展示单个候选的调用方式，正式运行把其余候选都加入同一个`--candidates`列表：

```bash
python scripts/blind_review.py prepare \
  --reference runs/main19_v1/generation/clean_dpo.json \
  --candidates runs/main19_v1/generation/eps_2_rr_42_lambda_0.5.json \
  --output-dir runs/main19_v1/human_review
```

评审者只收到`review.csv`，不要收到`private_key.json`。按相关性、正确性、帮助程度评判，不因更长就判更好。vote填写`left`、`right`、`tie`或`skip`，notes记录简短理由。建议两位成员独立复核至少20%的任务，先保留各自原始判断，再记录分歧处理；若使用LLM评审，另记模型、提示和版本，不能冒称人工判断。

```bash
python scripts/blind_review.py score \
  --votes runs/main19_v1/human_review/review.csv \
  --key runs/main19_v1/human_review/private_key.json \
  --output-json runs/main19_v1/human_review/scores.json
```

结果报告胜/平/负、skip数量和有效分母；平局按0.5计。空白或缺失任务会报错，不会悄悄忽略。

## 运行前检查与边界

- 本地CPU测试和小型随机Qwen端到端测试通过，不代表NPU/CUDA或真实1.5B训练已验证。
- 首先在服务器用独立目录做几步smoke test，确认设备、环境、显存和loss有限；不要用smoke输出替代主实验结果。
- 正式实验开始后保持代码版本、底座和配置固定；需要更改就创建新版本suite。
- 目前没有自动评判模型效果，也没有任何真实19组训练结果。
