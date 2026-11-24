# GLM-5.4 与 Loop 的能力来源

2026-11-24。本文将官方披露、数学推导和机制假说分开。来源见 [参考索引](../references/README.md)，尤其 [G1,G2]；共同抽象见 [01](01_mathematical_correspondences.md)。

## 1. 工业实例：官方披露与原图核对

[中文技术博文](https://www.zhipuai.cn/zh/research/164)（页面标注 2026-11-22 16:00；英文入口 [Z.ai](https://z.ai/blog/glm-5.4)）给出 GLM-5.4 架构与规模验证。2026-11-24 核对中文正文、[架构原图](https://www.zhipuai.cn/api/media/file/c1b61892-d25a-461d-8789-31315d77de15.png) 与 [消融原图](https://www.zhipuai.cn/api/media/file/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20261123163301_500_40.jpg)。此次整理未取得独立完整 5.4 技术报告或训练实现；[原 GLM-5 报告](https://arxiv.org/abs/2602.15763) 是基座参考。

### 1.1 结构

| 部分 | 不同 block 数 | 调用 |
|---|---:|---|
| Prelude L1–26 | 26 | 一次 |
| 中间核心 L27–54 | 28 | 两轮、主体共享 |
| Coda L55–78 | 24 | 一次 |

所以共有 78 个不同 block，执行 106 次 block 调用（不等于 106 层独立参数模型的 FLOPs、显存或能力）。此外有阶段独立 gate 等接口。官方从已经预训练的 GLM-5 基座继续训练，逐级考察 1B、4B、30B、750B 模型；官方称等预训练计算量下从头共享训练仍不如不共享模型，这也是选择后训练改造的动机。

三个重要接口（均来自官方）：

1. 独立逐 token damp gate 调节两轮候选变化的写回。第一轮总体较温和，第二轮更充分；首轮写入随 token 难度变化。门值不是实际跳过计算的信号。
2. 第一轮各层保存 KV latent 与 top-k 索引（架构图 k=2048）；第二轮用当前 query 读取第一轮稀疏 prefix 与本轮局部 SWA KV（图示 128 token），允许同位置两路表示共同参与一次 softmax。节省额外全上下文 KV。
3. 第二轮无独立 indexer，复用第一轮索引。官方报告与重新计算索引效果近似。

### 1.2 30B 受控结果

原图：与 750B 基座同架构的 30B 对照，报告同参数量、整体训练计算约 +10%，200K 长文本 SFT 后开启 thinking 评估。

| 基准 | 基线 | +Loop | 平均长度变化 |
|---|---:|---:|---|
| AIME25 | 19.17 | 30.42 | -635 tokens (-2.0%) |
| GPQA-Diamond | 47.58 | 56.26 | -2473 tokens (-11.7%) |
| AA-LCR | 40.33 | 47.67 | -539 tokens (-9.7%) |
| MMLU-Pro | 71.65 | 75.05 | -863 tokens (-14.9%) |
| IFBench | 50.20 | 53.47 | -763 tokens (-13.0%) |

这是厂商报告的 30B 结果，不是 750B 的单独 loop 消融。训练总预算增幅不是单次前向增幅；输出长度缩短与部分计算转移到隐空间的解释相符，但不能独立鉴别额外深度、门控、跨轮 KV、继续训练各自贡献。原文还提到旗舰的长程推理行为和部署速度；不把不同服务配置的吞吐当作受控硬件比较。完整训练数据、冻结策略、门控初始化、跨轮 KV 梯度、循环梯度截断等未公开，不能据此还原训练实现。

## 2. 相同权重为什么能执行不同内容？

设隐状态 $H^{(r)}_{ia}$，$i,j$ 为 token、$a,b$ 为隐藏通道、$\alpha$ 为 query/key 通道（省略 batch/head）。即使权重固定，
$$Q^{(r)}_{i\alpha}=\sum_a H^{(r)}_{ia}W^Q_{a\alpha},\quad K^{(r)}_{j\alpha}=\sum_a H^{(r)}_{ja}W^K_{a\alpha},$$
$$A^{(r)}_{ij}=\operatorname{softmax}_j\big((\sum_\alpha Q^{(r)}_{i\alpha}K^{(r)}_{j\alpha})/\sqrt{d_q}+M_{ij}\big).$$
状态变化导致 query、注意力路由、写入内容变化。FFN $f(h)=W_2\sigma(W_1h)$ 的局部 Jacobian
$$J_f(h)=W_2\operatorname{diag}(\sigma'(W_1h))W_1$$
也随状态改变；GLM 实际 MoE/MLA/稀疏注意力需要用真实路由图，公式仅用于说明算子机制。共享权重并不意味着共享激活或完全相同的运算分支。

| | Looped LM | 典型连续 diffusion/flow |
|---|---|---|
| 迭代状态 | 隐计算/工作记忆 | 噪声样本/生成 latent，可能带求解器历史 |
| 阶段 | 计算轮次或进度 | 噪声时间/概率路径 |
| 本次网络输出 | 隐状态，最终读出 token logits | score、噪声、速度或干净样本估计 |
| 更新 | 循环块+门控+状态接口 | 预测器+采样器 |
| 中间训练状态 | 模型自身 rollout | 标准训练可直接从已知前向路径抽样 |
| 中间目标 | 任务损失、可选深监督 | 外部构造的局部统计目标 |

普通自回归 LM 已在 token 轴共享权重；循环深度在每个 token 的隐计算增加另一条复用轴。

## 3. 训练：信用如何分配给共享权重的各次调用

设 $h_{r+1}=F_\theta(h_r,c)$、$y=R(h_K)$、任务损失 $L=-\log p(y^\star\mid c)$。完整反传时，固定不含共享参数的 Prelude 和读出，定义 $A_r=D_h F_\theta(h_r,c)$、$B_r=D_\theta F_\theta(h_r,c)$、$g_K=\nabla_{h_K}L$：
$$\nabla_\theta L=\sum_{r=0}^{K-1} B_r^\top A_{r+1}^\top\cdots A_{K-1}^\top g_K.$$
两轮为 $B_1^\top g_2+B_0^\top A_1^\top g_2$。后一项使第一轮为后续调用建立有用状态，前一项直接训练最后一次计算。参数共享聚合不同位置的任务要求。若使用截断 BPTT 或 detach，明确删去相应路径；这不是 GLM 实际反传策略的断言。Huginn [L2] 是随机循环预算及截断训练的先例。

标准 diffusion 噪声预测：
$$L_{\rm diff}=\mathbb E_{x_0,t,\epsilon}w(t)\|\epsilon_\theta(\alpha_t x_0+\sigma_t\epsilon,t,c)-\epsilon\|^2.$$
在固定前向扰动过程下 $D_\theta x_t=0$，单次梯度为
$$2\mathbb E[w(t)(D_\theta\epsilon_\theta)^\top(\epsilon_\theta-\epsilon)].$$
共享预测器适配不同时间的局部统计任务，但训练通常不展开完整采样链 [D1]。Loop 则可以用终点损失塑造模型自产中间状态。两种训练原则可组合：Loop 可引入局部监督或教师区间；diffusion 可展开或蒸馏完整轨迹。

## 4. 能力提升的可解释构造

### 4.1 复合已有关系

若共享检索算子 $F(e_i)=e_{f(i)}$，上下文中 $A\mapsto B\mapsto C$，则 $F(e_A)=e_B$，$F^2(e_A)=e_C$。第二次查询使用前次才得到的变量。全局 attention 一轮即可看到所有位置，但内容依赖的关系复合仍可能需要连续计算。它对应算法深度，而不是空间感受野。

对于迭代线性求解 $z_{r+1}=z_r+B_\theta(b-Az_r)$，从零出发，$z_1=B_\theta b$，$z_2=2B_\theta b-B_\theta AB_\theta b$。同一修正规则产生新的复合项。[L4,L5] 的合成算法、ICL 数据拟合实验支持共享块可以学到迭代处理；真实旗舰的通用推理机制还需干预证据。

### 4.2 参数预算与执行预算

$P$ 参数的 $k$ 层模块循环 $R$ 次，主体参数仍约 $P$，深度 $kR$；不共享同深模型约 $RP$。结构匹配时 $\mathcal F_{\rm shared}\subseteq\mathcal F_{\rm unshared}$，因为独立层可以取相同参数。Loop 的优势来自固定存储预算下执行更深，以及重复规则作为归纳偏置；不是同等深度时无条件具有更大的函数类。

### 4.3 为什么改造后需要训练

预训练模块输入分布 $\mu_{\rm in}$，输出分布 $G_\theta{}_\#\mu_{\rm in}$，循环使共享核心接收改变后的分布。后续训练让各轮协作、门控稳定尺度、Coda 适配新状态。区分已有规则重复组合与新训练形成的协作；需要同数据/同预算的额外继续训练基线。

## 5. GLM 的信息通路给出的具体机制

官方门控可写为
$$X_{r+1}=X_r+\beta_r(X_r)\odot(G_\theta(X_r,M_r)-X_r),$$
其中 $\beta_{r,i}=\operatorname{sigmoid}(w_r^\top\operatorname{RMSNorm}(X_{r,i})+b_r)$（简化写法）。每 token 标量门局部 Jacobian：
$$JT=(1-\beta)I+\beta JG+(G-x)\nabla\beta^\top.$$
完整 token 序列还包括交叉 token 和 memory 路径。

第二轮注意力以两路共同归一化：
$$O^{(2)}_i=\sum_{j\in I_i^{(1)}}a_{ij1}V_j^{(1)}+\sum_{j\in W_i}a_{ij2}V_j^{(2)}.$$
索引复用仍允许 query 和权重变化；局部新 KV 允许重新读取新表示。第一轮输出 $S_1=(X_1,M_1)$，第二轮依赖两者。第一轮 gate 小不意味着 memory 贡献小：
$$\delta H_2\simeq D_{X_1}T_2\delta X_1+D_{M_1}T_2\delta M_1.$$

可检验的三个机制假说：
- 第一轮建立关系，使第二轮 query 查找新的任务相关信息。
- 第一轮建立被第二轮读取的上下文表示；未来任务损失可以训练这种写入（如果实现保留该梯度）。
- 第二轮复用中间变量继续关系复合或约束校正，形成更少显式 token 下的更好结果。

下一步应做匹配计算的干预：固定索引、分别交换 query / 两路 KV / 主状态与门控；观察终点 logits、任务正确率与生成长度。详见 [04](04_research_questions.md)。