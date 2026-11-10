# 正文核实笔记

以下定位对应[sources.json](sources.json)记录的PDF版本。结论按论文报告整理，未独立复现实验。用章节/公式定位，避免不同排版的页码混淆。

## 1.DPO：训练目标与生成评估的起点

[原文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)。重点：§4、Eq.(7)、§6。

- 从KL正则奖励最大化和BT偏好模型得到logistic目标，margin使用策略相对固定参考策略的log概率差。
- §6分别做IMDb情感生成、TL;DR摘要、HH对话；真实任务使用生成胜率，不能用训练loss下降替代。
- 对项目的要求：训练/评估共用编码与参考模型；干净DPO是参照，生成盲评是必要补充。论文没有提供隐私保证。

## 2.rDPO：完全校正的来源，也是“部分校正”查新的关键

[原文](https://proceedings.mlr.press/v235/ray-chowdhury24a.html)。重点：§2、Eq.(9)–(13)、Lemma3.1、§4、§6.2。

- Eq.(9)–(10)先写含噪观测似然；Eq.(11)是cDPO平滑；Eq.(12)才是减去反向损失的无偏校正。三者不能混称。
- §2明确实际可交叉验证翻转率；§3讨论方差。§4保证依赖参数化、有界性及覆盖等条件，不能直接外推到任意神经网络优化轨迹。
- §6.2在HH上尝试多个校正率，再比较生成；因此“扫描λ”和“准确率+生成”都不是首创。
- 我们的贡献空间在有限数据、实际隐私接口和选择可靠性。当前λ等价关系见[推导](INNOVATION.md#部分校正究竟是什么)。

## 3.Private Reward Estimation：保护范围与噪声代价

[原文](https://proceedings.mlr.press/v238/ray-chowdhury24a.html)。重点：Definition2.2–2.3、§4–6。

- 固定公开状态/动作特征，只改变偏好标签；local机制在交给分析者前扰动，central模型允许可信持有者处理原标签。
- 局部RR与中心化目标扰动的估计误差代价不同。§6采用低维BTL合成数据、不同样本量与隐私预算，不是LLM生成结果。
- 项目应引用它解释label-DP和RR，不借它宣称1.5B模型达到最优效用。网页摘要与下载PDF对central维度因子的写法不同，本笔记不引用该精确速率。

## 4.Unified Analysis：RR+rDPO已经被直接研究

[原文](https://proceedings.mlr.press/v267/zhou25ac.html)。重点：Definition3.4、Algorithm1、Theorem6.3、AppendixD。

- 区分CTL与LTC，线性模型下后者可能有更大的污染代价；随机隐私噪声与对抗污染不是同一因素。
- AppendixD使用1697条GPT-4o合成金融偏好、GPT2-large、85/5/10划分，明确把RR翻转率设置为1/(exp(ε)+1)，再训练rDPO；生成裁判为llama3:70b，报告五个种子。
- 这直接排除了“首次将RR用于rDPO”的表述。我们的HH/LoRA、多RR受控比较是不同实证条件，不自动构成算法创新。
- 现有19组没有另加对抗污染，不能用于验证CTL/LTC分离结论。

## 5.SquareχPO：平方的是概率，不是普通DPO的margin

[作者PDF](https://xingyuzhou.org/publications/TPDP-square.pdf)。重点：第3页Algorithm1、§3.1、第4页Assumptions3.2–3.3与Theorem3.5；公式已对照渲染页面。

- 使用φ(u)=u+log(u)的χPO参数化；先截断隐含奖励差、过sigmoid，再对概率变换与缩放标签做平方损失。
- 一般函数逼近结论仍要求策略可实现性、有界奖励差等条件，并非无条件适用于任意LoRA模型。
- 本次所读版本提供理论与证明，没有LLM实验章节。出版场所未确证，仅标作者版。
- 借鉴点是研究稳定目标；不能把我们实现中的logistic loss换成margin平方，就称已实现SquareχPO。

## 6.PrivDPO：同样保护偏好，但发布接口不同

[原文](https://arxiv.org/abs/2608.05040)。重点：Definition3.1–3.3、Algorithm1、Theorem4.1后的Remark、§5。

- 保护候选文本固定时的偏好翻转，利用梯度差所在的偏好轴，随机缩放目标并保持梯度无偏。
- Algorithm1对不相交批次做一次遍历；Remark明确所有训练实验使用一轮。重复访问同一偏好要组合隐私损失。
- §5涵盖Qwen2.5、Llama3、Pythia，主要单随机种子；不能据此宣称我们的多RR设置已被覆盖或其方法一定优于我们。
- 与我们的区别：我们可先在客户端固定发布RR标签；其训练机制需要偏好相关计算。比较必须说明可信计算边界、轮数与总隐私预算。
- arXiv注明“CCS2026接收的扩展版”；本次核实到作者声明，未独立确认ACM正式版本。

## 7.DPPrefSyn：隐私文本需要另一条路线

[原文](https://proceedings.mlr.press/v306/gao26ae.html)。重点：Definition3.1、Algorithm1、§5、AppendixA；算法页已对照渲染页面。

- 私有记录经过DP降维、聚类和奖励模型学习，再给公开prompt的生成候选排序，产生可复用的合成偏好。
- §5覆盖OpenAssistant、HH和TL;DR；这是中心化整条记录(ε,δ)-DP路线，不是客户端单标签纯ε-RR。
- **需进一步审计的细节：**Algorithm1第11–12行直接从簇大小生成抽样权重，正文称其为DP histogram；AppendixA未单列这一计数步骤的预算。复用前须追查DP-KMeans接口是否已提供受保护计数，以及数据依赖分簇的会计。此处标为未解决疑点，不据此断言论文错误。
- 当前项目不扩展到合成数据，否则要重新设计信任模型和实验预算。

## 8.AUP-RLHF：单条隐私不等于用户隐私

[COLM官方条目](https://proceedings.colm.cc/paper_files/paper/2025/hash/1df4afb0b4ebf492a41218ce16b6d8df-Abstract-Conference.html)；阅读正文为[arXiv版本](https://arxiv.org/abs/2502.17515)。重点：Definition2、Algorithm1–2、§6.1。

- 固定公开文本，邻接数据可改变同一用户的整组偏好。简单把用户预算分给每条RR会损失效用，论文改进用户级奖励学习。
- 实验用Gemma-2-2B与LLaMA-2-7B，情感/摘要任务；设置每用户贡献m=10或50，不应当作我们的单条RR实验。
- HH当前划分没有可依赖的用户贡献分组。真实落地必须限制贡献或做用户级会计，不能仅把“record”改写成“user”。

## 补充查新：已核实摘要/入口，未计入8篇精读

- [Patrini等，CVPR2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html)：forward/backward loss correction早已有系统研究。换一种噪声校正写法不够新。
- [Chen等，Robustness of Accuracy Metric，AAAI2021](https://arxiv.org/abs/2012.04193)：摘要明确讨论带噪验证集用于超参数和早停选择。不能把“无需干净验证标签”本身声称为首创。
- [Papernot与Steinke，Hyperparameter Tuning with Rényi Differential Privacy](https://arxiv.org/abs/2110.03620)：调参的信息访问也要纳入隐私。它分析的重复私有训练情形，不等于固定RR数据上的纯后处理。

查新覆盖：上述论文引用链、2024–2026偏好隐私/鲁棒对齐，以及loss correction、noisy validation、private hyperparameter tuning交叉检索。没有穷尽全部论文，也不能据此证明任何方案“世界首次”。
