# 8篇核心论文比较表

本项目范围：公开prompt与候选回答，保护单条二元偏好；固定RR发布后训练Qwen2.5-1.5B-Instruct+LoRA。ε是隐私预算，ρ=1/(1+exp(ε))是翻转率，λ是校正强度。论文中的同名符号未必同义，尤其rDPO原文ε表示翻转率。

|论文及原文|保护对象与信任假设|方法、证据与限制|与本项目的关系|
|---|---|---|---|
|[DPO，NeurIPS2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)|无DP保证|用策略/参考策略概率比表达奖励；IMDb、摘要和HH生成实验；生成胜率与偏好排序准确率是不同指标|干净DPO是非隐私参照；RR-DPO才是同隐私条件基线。支持我们补生成盲评|
|[rDPO，ICML2024](https://proceedings.mlr.press/v235/ray-chowdhury24a.html)|处理随机翻转；算法本身不负责提供DP|逆向损失校正无偏；理论依赖策略参数化与覆盖条件；IMDb及HH实验|当前完全校正的直接来源。原文已讨论方差、调校正噪声率；部分校正不能包装成全新损失|
|[Private Reward Estimation，AISTATS2024](https://proceedings.mlr.press/v238/ray-chowdhury24a.html)|公开特征、私有标签；分别分析local和central label-DP|BTL奖励估计；RR与中心化扰动；证据主要是线性理论和合成数据|解释RR隐私基础及噪声代价；不直接证明我们的非线性LoRA训练更好|
|[Unified Analysis，ICML2025](https://proceedings.mlr.press/v267/zhou25ac.html)|local label-DP与对抗标签污染；区分污染/隐私先后顺序|统一RLHF/DPO理论；附录以GPT2-large和合成金融偏好实测RR+rDPO|**最直接的先例**：RR+rDPO不是新组合。我们的差异只能来自明确的新问题与证据|
|[SquareχPO，作者版](https://xingyuzhou.org/publications/TPDP-square.pdf)|local label-DP+Huber污染|χPO参数化、概率平方损失、标签缩放及截断；一般函数类理论仍有可实现性/有界条件；所读版本无LLM实证章节|说明替代损失已有理论路线。不是给DPO的margin简单平方就能复现；不列作已复现基线|
|[PrivDPO，2026预印本扩展版](https://arxiv.org/abs/2608.05040)|固定公开文本、单条偏好相邻；保护训练产生的随机梯度机制输出|沿偏好轴随机缩放目标，避免逐样本梯度操作；三类模型/三任务，主要单种子；单轮协议|重要近期比较对象，但不是客户端先发布一次RR标签的相同接口。多轮需要重新会计，不可直接横比ε和训练轮数|
|[DPPrefSyn，ICML2026](https://proceedings.mlr.press/v306/gao26ae.html)|中心化(ε,δ)-DP，邻接单位为整条偏好记录|DP-PCA、DP聚类、DP奖励模型，再用公开prompt合成偏好；QA/摘要实验|回答“prompt也私密怎么办”。保护范围和流程不同，不能与我们的纯ε-label-LDP按一个ε排榜|
|[AUP-RLHF，COLM2025](https://proceedings.colm.cc/paper_files/paper/2025/hash/1df4afb0b4ebf492a41218ce16b6d8df-Abstract-Conference.html)|公开文本，保护同一用户的一组偏好；user-level label-DP|自适应用户级私有奖励学习；情感/摘要实验；不是直接DPO|真实用户通常贡献多条，单条RR不自动等于用户级保护。现有HH划分缺少可靠用户分组，不能宣称已验证用户级DP|

## 比较时必须对齐什么

- **保护谁：**单标签、整条记录、同一用户全部记录不是同一个邻接关系。
- **谁看过原标签：**客户端RR、可信服务器中的DP训练、只保护最终模型不能混用。
- **发布了什么：**固定一份RR数据的多轮后处理，不等于反复访问原标签或发布多份RR数据。
- **验证用什么：**干净验证是研究参照；部署若只有RR标签，选模型也必须遵守这个信息边界。
- **指标是什么：**分类排序、生成质量、训练成本分别报告。不同数据/模型/裁判的论文数字不拼成统一排行榜。

## 对我们现有设计的直接影响

保留19组作为受控复现实验。把λ=.5称为“欠校正设定”，检验它是否有用，不预设必胜；增加无干净验证标签的选择评估作为实践问题。若要声称方法优于既有校正，优先补forward correction，而不是只比较λ=0与1。具体预算和判定规则见[创新设计](INNOVATION.md)。
