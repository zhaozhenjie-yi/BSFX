# 版本、复现范围和未完成项

## 目录

`src/chapter4_repro` 保留当前主干与运行依赖：HPE、GRU/CNN、DWAF、投影、结构 Transformer 和分类器，以及历史 Paper/Engineering profile。当前已训练主线的上下文代码在 `src/context_model.py`、`src/semantic_encoder.py`、`src/semantic_encoder_v3.py`；V2/V3 上下文版本与 Paper/Engineering 主干路线是两个不同维度。

`experiments/supervised_training`、`component_ablation`、`relation_ablation` 是对应训练/评估脚本快照。`experimental/return_history` 保留尚未完成学习确认的返回关系、历史 guard 扩展。

## 当前设置

XC2263N Full 使用 Engineering 主干、V2 上下文、自然采样、加权交叉熵、学习率 1e-4、Dropout 0.1、残差缩放 0.1、800 步。主干为 512 维、六层八头 Transformer。各消融实际设置以 `SCIENTIFIC_CONFIG.json` 和聚合结果中的 `arm` 为准。

三个 seed 共用同一份初始权重，改变训练随机过程和采样顺序。共同初始文件 SHA 与代码 SHA 保存在 `provenance/CODE_VERSION.json`，权重文件未公开。

阈值选择保留固定 0.5 与验证校准两种结果。最终校准使用验证分数分界点候选，以 Macro-F1、balanced accuracy 和距 0.5 的距离顺序选择；历史早期脚本仍可能含有限网格版本，以最终评估脚本和公开数值结果为准。阈值没有在本次发布过程中重新对测试集优化。

## 公开包可验证的内容

`python tools/verify_aggregates.py` 只用标准库，检查源码 SHA、Python 语法、JSON、各混淆矩阵及分类指标的一致性和聚合均值。它不运行模型，也无法仅凭聚合表重新计算 ROC-AUC、AP 或 CE。

这些排序/概率指标已在发布前对本地保存的逐样本预测进行独立算术核验，范围记录于 `PRIVATE_EVIDENCE_CHECK_SUMMARY.json`。本次未重跑全部历史实验；历史摘要和当前 800 步结果分别标明。

## 完整复现实验还需要什么

原始数据及其授权、预处理缓存、词表、初始权重和软件依赖环境需另行准备。代码中的 `BSFX_WORKSPACE`、`BSFX_ENV`、`EXTERNAL_LOCATION` 等是位置占位符，不是真实机器信息，也不是可直接执行的部署配置。公开目录对源码进行了归类，历史脚本的搜索路径需要按新部署重新映射；本次未验证该迁移后的训练执行。

没有凭空添加一个声称可以完全恢复历史环境的依赖锁文件。原始源码/公开源码的哈希分别记录，算法没有根据测试分数再调优。

## 结论边界

当前报告标签不等于真实漏洞真值；MISRA/ISO 注释代理不等于生产安全标签；机制契约也只适用于规定操作与输入。当前数据不足以证明 V3 稳定优于 V2、每种关系都有独立收益，或模型已经通过跨函数关系识别真实跨函数漏洞。

源文件和历史数据暴露限制继续保留。后续若改数据、模型或阈值，最终确认需要新的独立来源集合。返回关系与历史 guard 分支已经完成表示回归、兼容性及梯度检查，但尚未完成正式学习与泛化验收。
