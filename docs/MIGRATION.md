# Report03迁移说明

原始压缩包保持不变。本目录只保留新版必要源码，历史脚本、旧评估JSON和旧报告继续保存在原包，避免被误认为新版证据。

## 目标函数

令z为按观测标签排序的β缩放policy-reference margin，ρ=1/(1+exp(ε))：

```
L_plain(z) = softplus(-z)
L_corrected(z) = ((1-rho)*softplus(-z) - rho*softplus(z)) / (1-2*rho)
L_partial(z) = (1-lambda)*L_plain(z) + lambda*L_corrected(z)
```

`objective.partial_loss`显式实现这一公式。λ与隐私预算独立。只对固定的已扰动数据做训练，不在每轮重新访问干净标签。提供的RR种子用于公开数据复现。

## 数据接口

保留原项目JSONL字段`id`、`prompt_id`、`prompt`、`response_a`、`response_b`、`label`。label=1表示B优于A，label=0表示A优于B。新RR文件附带`.meta.json`；旧RR文件没有该元数据，不能直接假装成新版RR数据，需要重新生成。

多轮解析只识别双换行之后的Human/Assistant角色标记，容许角色名前空格。连续同角色消息合并，并记录合并数量。正文内角色字样保留为正文，不自动推测缺失的换行。chat模式去除各段首尾空白；raw_hh保持原字符串。

## 训练环境

原实验使用TRL内部预处理。本版用独立共享编码器和轻量Trainer子类，让训练和评估实际消费同一份input_ids、attention_mask、completion_mask。参考模型通过禁用LoRA得到；底座冻结，dropout关闭。保留LoRA默认rank16、alpha32、attention/MLP目标模块、学习率1e-4、β=0.1、batch4、3轮、cosine和5% warmup。

新格式、dropout与RR随机流变化意味着必须重新训练新版基线，不能只替换一组后混用旧结果。raw_hh可用于旧模型格式诊断；它不承诺精确复制旧TRL版本对跨token边界等特殊样本的全部处理，遇到边界不一致会报错。
