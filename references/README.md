# 文献与技术资料索引

研究快照：2026-11-24。编号供主体文档引用；链接优先指向作者原文、会议页或官方发布。记录的是研究相关机制，不以预印本等同于同行评审。标明最新一轮核对的重点；其他条目沿用前序调研的原文入口，深入复现前仍应固定论文版本、代码 commit 和配置。

## G. 工业实例

| ID | 原始资料 | 研究作用 |
|---|---|---|
| G1 | [GLM-5.4 中文技术博文](https://www.zhipuai.cn/zh/research/164)；[英文入口](https://z.ai/blog/glm-5.4) | 预训练后改造、28 层两轮、阶段 gate、双路 KV、索引复用、30B 消融；中文正文和两张图已于 11-24 重核 |
| G2 | [GLM-5 技术报告](https://arxiv.org/abs/2602.15763) | 基座结构参考，不包含 5.4 新的循环训练细节 |

G1 原图：[架构](https://www.zhipuai.cn/api/media/file/c1b61892-d25a-461d-8789-31315d77de15.png)、[30B 消融](https://www.zhipuai.cn/api/media/file/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20261123163301_500_40.jpg)。中文页面标注 2026-11-22；原图数字见 [GLM 专章](../docs/02_glm_and_learning_mechanisms.md)。未取得独立完整 5.4 论文、公开训练代码与训练梯度边界。

## L. 循环深度与隐推理

| ID | 原文 | 可迁移认识 |
|---|---|---|
| L1 | [Universal Transformers](https://arxiv.org/abs/1807.03819) | 共享深度模块、阶段编码、动态计算 |
| L2 | [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)（Huginn） | 随机隐态初始化、输入注入、随机预算和截断反传 |
| L3 | [LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation](https://arxiv.org/abs/2602.11451v1)（ICLR 2026） | 时间/步长条件、长短轨迹一致性；已直接迁移 diffusion shortcut 思想 |
| L4 | [Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416) | 同参数浅层、同深度非共享模型对照；合成多步任务 |
| L5 | [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424) | ICL 数据拟合与迭代求解的经验 |
| L6 | [Loop-μP: Residual Scaling and Hyperparameter Transfer in Looped Transformers](https://arxiv.org/abs/2606.18524v3) | 特定理论条件下跨轮残差相关、1/K 缩放、学习率迁移；11-24 重核 v3 摘要 |
| L7 | [Stabilizing Equilibrium Models by Jacobian Regularization](https://proceedings.mlr.press/v139/bai21b.html) | Jacobian 估计与平衡稳定性；有限步任务需采用任务相关传播度量 |

## D. Diffusion、flow 与训练经验

| ID | 原文 | 可迁移认识 |
|---|---|---|
| D1 | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 外部定义前向扰动、随机时间局部回归 |
| D2 | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | 确定性更新、门控重参数化起点 |
| D3 | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | SDE、概率流 ODE、边缘分布与路径 |
| D4 | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | 概率路径、速度场和连续性方程 |
| D5 | [Consistency Models](https://arxiv.org/abs/2303.01469) | 沿同轨迹预测共同终点 |
| D6 | [Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion](https://arxiv.org/abs/2310.02279) | 区间映射、组合、共同终点比较 |
| D7 | [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)（EDM） | 输入、输出、跳连和噪声阶段预条件 |
| D8 | [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304) | timestep-aware PTQ，跨阶段激活校准 |
| D9 | [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/abs/2208.04202) | 将自身先前预测作为后续输入 |
| D10 | [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009) | 自生成历史和训练-推理状态分布 |
| D11 | [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556) | 多时间任务梯度冲突与损失加权 |
| D12 | [Self Gradient Forcing: Native Long Video Extrapolation](https://arxiv.org/abs/2607.20368v2) | 两阶段重构恢复未来读到历史 KV 的梯度，历史 latent detach；11-24 重核 |
| D13 | [dKV Forcing: Separating Generation and Memory Representations through a Differentiable KV Cache](https://arxiv.org/abs/2611.01648v2) | 可微缓存、生成与写入角色冲突、低秩写入接口；11-24 重核 |
| D14 | [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927) | 利用已定义 diffusion ODE 结构的高阶求解；不同于任意离散 Loop |

精确高斯后验均值的 score 与 covariance-Jacobian 恒等式可直接对高斯似然积分微分推得（主体文档 01）；使用前提是相应正则性及有限二阶矩，不依赖某个学习模型的实测结论。

## X. 直接联系、交叉模型与新近工作

| ID | 原文或官方入口 | 与本研究的联系 |
|---|---|---|
| X1 | [Looped Diffusion Transformer](https://arxiv.org/abs/2609.40305)，[项目页](https://looped-dit.github.io/) | 内层共享中间块、深监督、自调制 attention，联合内外层预算；11-24 重核摘要及项目页 |
| X2 | [LiFT: Loop Flow Transformers](https://arxiv.org/abs/2610.05538) | 连续深度监督、初始估计到目标速度的校正路径，循环外推；11-24 重核检索摘要 |
| X3 | [Deep Equilibrium Approaches to Diffusion Models](https://arxiv.org/abs/2210.12867)（NeurIPS 2022） | 整条 DDIM 链的联合固定点，并行求解与反演 |
| X4 | [One-Step Diffusion Distillation via Deep Equilibrium Models / GET 官方实现](https://github.com/locuslab/get) | 将 teacher 噪声-样本映射蒸馏到平衡生成器，固定噪声注入与隐态求解 |
| X5 | [Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models](https://proceedings.mlr.press/v306/geiping26a.html)（ICML 2026），[arXiv](https://arxiv.org/abs/2510.14961) | diffusion forcing 调度直接迁移，跨 token 并行精炼；已重核会议摘要 |
| X6 | [On the Reasoning Abilities of Masked Diffusion Language Models](https://openreview.net/forum?id=BVnIsh4Nz1)（ICLR 2026） | 特定精度、宽度和工作空间下的形式化模拟关系 |
| X7 | [LoopPrune: Recurrent Pruning for Looped Diffusion Transformers](https://arxiv.org/abs/2611.22757)（2026-11-17） | 直接关联共享参数、循环位置与剪枝恢复；11-24 新发现、摘要级核对，优先全文查重 |

## 迁移与新颖性边界

- DDIM 门控重参数化、终点一致性与轨迹固定点来自已有数学结构，不作为新颖性声明。
- LoopFormer 已解决一类可变预算与长短路径对齐；Looped-DiT、LiFT 已探索内部循环和外部采样的计算分配。
- SGF、dKV 已研究未来损失到历史 KV 写入，Loop 中的直接应用尚待验证。新颖性需要体现不同状态拓扑、梯度边界或真实任务收益。
- Loop-μP 已推导共享残差跨轮相关的缩放；LoopPrune 已考虑共享移除与循环位置的重要性。后续压缩工作需与它们明确区分，先研究完整执行敏感度的交叉项能否预测真实任务损失。
- 文献的作者报告与我们的模型实验分别记录。后续建立复现时再登记精确 commit、环境、模型权重、数据与命令；不将尚未检查的实现路径写成已复现。
