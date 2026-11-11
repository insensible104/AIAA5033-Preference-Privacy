# 19+6组实验与带噪验证选择

更新日期：2026-11-11。代码和配置已补齐；本次没有启动训练或模型评估。现有19组及固定最终adapter的主结果口径保持不变。新增6组forward、12候选选择实验单独报告。

## 固定设计

|项目|设定|
|---|---|
|主实验|原19组，见[原协议](RUN_MAIN19.md)|
|新增训练|ε∈{1,2}×训练RR种子∈{42,43,44}，共6组forward|
|配对要求|继承主实验底座、revision、训练dtype、训练参数、划分、训练种子；直接复用主实验RR文件，不重新加噪|
|forward目标|−log[ρ+(1−2ρ)σ(z)]，代码用log-sigmoid+logaddexp计算；ρ是机制翻转率，不调λ|
|选择候选|每个(ε,训练RR种子)独立选择：RR-DPO、λ=.5、rDPO、forward，各取checkpoint-1250、checkpoint-2500、最终adapter，共12个|
|默认模型|同条件RR-DPO最终adapter|
|验证预算|默认ε_val=ε_train；在配置中单独记录|
|验证量|固定嵌套子集100/250/500；同一完整RR发布取子集|
|验证RR重复|20次，仅用于公开数据模拟；同预算的所有候选和三个训练RR种子共享每次验证发布|
|选择规则|带噪准确率argmax；配对Hoeffding下界大于0才替换默认模型；四个固定最终模型|
|离线参照|干净验证argmax，在带噪选择冻结后独立计算，测试前冻结|
|并列|按配置登记顺序，先family再endpoint；不临时挑选|

配置：[forward6.json](../configs/forward6.json)、[6组清单](../configs/forward6_matrix.csv)、[selection.json](../configs/selection.json)。forward的`correction=0`仅为旧manifest字段兼容，实际通过`group=forward`和`objective=forward`分派，绝不是RR-DPO的λ=0损失。

加入forward后，初稿的9个候选变成12个。置信半径使用实际K=12：n=500、α=.05时，ε_val=1/2对应约33.77/20.49个百分点。保守规则可能总是回退，必须如实报告。

## 1.只准备配置与打印命令，不启动训练

在仓库根目录执行。若已生成尚未访问测试集的主suite，复用它，跳过第一条plan。每个新输出目录必须为空。

```bash
python scripts/run_main19.py plan \
  --suite runs/main19_v1 --model Qwen/Qwen2.5-1.5B-Instruct --dtype bfloat16

python scripts/run_main19.py plan \
  --suite runs/forward6_v1 --config configs/forward6.json \
  --paired-suite runs/main19_v1

python scripts/run_selection.py plan \
  --output-dir runs/selection12_v1 \
  --main-suite runs/main19_v1 --forward-suite runs/forward6_v1 \
  --config configs/selection.json

python scripts/run_main19.py train --suite runs/main19_v1 --device npu
python scripts/run_main19.py train --suite runs/forward6_v1 --device npu
python scripts/run_selection.py predict --output-dir runs/selection12_v1 --device npu
```

这些命令只准备数据/manifest或打印执行命令。远程模型plan会查询revision元数据，不下载模型权重。forward继承主suite的模型与dtype，不再传`--model`或`--revision`。

选择计划登记72个候选adapter身份位置、验证子集与RR规则，但不加载权重。验证RR与训练RR使用不同随机流。配置修改必须创建新研究目录，不能修改已冻结的plan。

## 2.将来获准运行时的顺序

以下是运行说明，不代表本次已执行。训练和模型推理只有加`--execute`才运行。`select`、`oracle`、`summarize`是轻量计算，调用即写输出，不需要此开关。

```bash
# 先完成25组训练。CUDA把npu换成cuda。
python scripts/run_main19.py train --suite runs/main19_v1 --device npu --execute
python scripts/run_main19.py train --suite runs/forward6_v1 --device npu --execute

# 72个候选只对规范顺序的公开验证文本计分；此时不提供真实偏好。
python scripts/run_selection.py predict --output-dir runs/selection12_v1 --device npu --execute

# 只读净化后的预测和RR标签，生成并冻结全部选择结果。
python scripts/run_selection.py select --output-dir runs/selection12_v1

# 独立的离线分析步骤：计算干净验证参照；先冻结，再访问测试。
python scripts/run_selection.py oracle --output-dir runs/selection12_v1

# 统一评估登记的72个候选，包含事后regret诊断所需的模型。
python scripts/run_selection.py test --output-dir runs/selection12_v1 --device npu --execute
python scripts/run_selection.py summarize --output-dir runs/selection12_v1
```

选择计划和选择冻结会拒绝源suite中已有的测试评估或生成输出，oracle也拒绝测试已访问的情况。这能拦截本流程中的误操作，不能证明操作者从未在别处看过测试结果。若已经查看测试，应明确标为探索性研究或另留测试集，不能删除文件伪装成盲测。

失败后可重跑predict/test：已存在结果会检查数据、配置、模型与adapter哈希，通过才跳过；不覆盖已有结果。若输出损坏，先调查原因，不直接把它当成成功。选择输出一旦冻结不能重选；不完整的选择构建应在未访问测试的前提下重新登记新目录。

## 3.仍单独报告固定最终模型的25组比较

在选择结果和干净验证参照都冻结后，按原流程生成主表及forward比较。以下验证评估使用原始干净方向，仅供离线诊断和固定最终模型报告，不输入带噪选择器。

```bash
python scripts/run_main19.py validation --suite runs/main19_v1 --device npu --execute
python scripts/run_main19.py validation --suite runs/forward6_v1 --device npu --execute
python scripts/run_main19.py freeze --suite runs/main19_v1 --execute
python scripts/run_main19.py freeze --suite runs/forward6_v1 --execute
python scripts/run_main19.py test --suite runs/main19_v1 --device npu --execute
python scripts/run_main19.py test --suite runs/forward6_v1 --device npu --execute

python scripts/summarize_main19.py --suite runs/main19_v1 --split test \
  --output-dir runs/main19_test_summary
python scripts/summarize_forward.py --main-suite runs/main19_v1 \
  --forward-suite runs/forward6_v1 --split test --output-dir runs/forward_test_summary
```

`summarize_forward.py`输出25个最终模型的逐组指标/可用成本，以及forward对λ=0/.5/1的18个同RR配对比较、跨3个训练RR种子的均值与SD。这里不使用选择器选出的中间checkpoint替换主结果。

当前选择评估和旧主表评估有部分重复推理，因为输入方向及输出身份不同；没有宣称零评估成本。先保留清晰的身份校验，不复用不兼容的旧JSON。

## 4.文件与信息隔离

|文件|用途|
|---|---|
|`plan.json`|事先登记来源suite哈希、候选列表、选择规则|
|`public_validation.jsonl`|仅公开文本；回答按与偏好无关的字典序排列，ID由公开内容生成；`label=0`只是A−B计分约定|
|`releases.json`|各预算/重复/子集的RR标签|
|`validation/*.json`|规范A−B方向的模型margin；其中`correct`仅相对公开计分方向，无干净准确率含义|
|`bundle.json`|仅保留候选预测、RR标签、顺序与规则；拒绝多余字段|
|`choices.json`、`selection_freeze.json`|选择结果及输入/预测哈希；冻结后修改会阻止测试|
|`simulation_private/clean_validation.json`|离线模拟者保留的干净标签映射，独立选择器不读取|
|`oracle.json`|测试前固定的干净验证参照|
|`test_release.json`|首次实际测试阶段锁定选择冻结与oracle身份|
|`summary/decisions.csv`|每次重复/子集/规则选了谁及测试表现|
|`summary/by_context.csv`|每个训练RR条件内，验证RR重复的均值、波动、选差比例和回退率|
|`summary/by_epsilon.csv`|先对验证RR重复取均值，再在3个训练RR条件间统计，避免把60次选择当60次训练|

可把`bundle.json`单独复制到不含干净数据的目录，运行纯选择器：

```bash
python scripts/select_noisy.py --bundle /path/to/bundle.json --output /path/to/choices.json
```

代码接口隔离不等于操作系统沙箱。真实部署还需要客户端可信随机化、权限与日志控制。本仓库是公开HH数据模拟；公开种子及反复生成RR发布不能直接照搬成真实隐私服务。

## 5.统计和隐私解释

- ties记0.5。相同ρ下去偏准确率与原始带噪准确率的argmax一致；输出去偏值用于解释，不称为新选择算法，不裁剪到[0,1]。
- 保守规则的半径为`√(2log(2(K−1)/α)/n)/(1−2ρ_val)`。α=.05是每个预登记选择问题的界，不是全部预算、子集、重复和训练种子的同时95%保证。
- 选择发生在固定候选集上；不能查看验证标签后新增候选。重复和嵌套子集共享样本，不构成独立测试集。
- `difference_vs_clean_selector_points`可为负；`hindsight_test_regret_points`是相对测试集内最佳候选的事后诊断，不能用来调参。
- 训练RR种子SD与验证RR重复SD分开报告；都不能代表训练随机种子的不确定性。
- 同一份已发布RR数据上的训练与选择是后处理；20次验证RR是公开数据模拟，不是20次免费私有发布。用户跨记录贡献需要另外做用户级会计。
- 本轮只增加排序层面的选择评估。选中中间checkpoint后，如果要声称回答质量改善，必须另补该checkpoint的生成盲评；现有8模型生成计划不自动覆盖它。

## 6.无需训练的验证

本次运行以下检查，明确跳过原有包含训练/优化器更新的3项测试：

```bash
pytest -q -k 'not train_save_reload_evaluate_and_protocol_guard and not plan_and_one_real_tiny_run and not reference_fixed_and_batch_invariance'
ruff check --isolated --select E4,E7,E9,F src scripts tests
```

新增测试覆盖forward干净极限、概率目标的总体最优点、极端margin数值/梯度、Trainer分派、规范顺序、独立选择接口、配对数据、dry-run、冻结与篡改拒绝、模拟评估汇总。模拟adapter文件只是身份占位，不是真实权重；不会调用模型加载或训练。NPU/CUDA上的实际训练与推理尚未验证。
