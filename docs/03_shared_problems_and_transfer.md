# 参数复用产生的共同问题与双向迁移

2026-11-24。使用 [01](01_mathematical_correspondences.md) 的完整状态转移和 [02](02_glm_and_learning_mechanisms.md) 的实际架构。参考 ID 见 [文献索引](../references/README.md)。

## 1. 问题图谱

| 旧问题 | 循环复用改变的条件 | 数学对象 | 已有经验 |
|---|---|---|---|
| 初始化、归一化、学习率迁移 | 残差增量跨轮相关 | 二阶矩、交叉项 | Loop-$\mu$P、EDM |
| 量化、剪枝、缓存近似 | 相同参数偏差进入每一轮 | 终点敏感度、交叉项 | Q-Diffusion、LoopPrune |
| 多阶段训练 | 模型访问自己产生的状态；共享任务冲突 | 占据分布、梯度 Gram | Self-conditioning、Self Forcing、Min-SNR |
| KV 与激活 | 保存的是未来会读取的工作状态 | 状态充分性、历史核、信用分配 | SGF、dKV Forcing |
| 深度选择、蒸馏 | 多轮可能延长算法或细化轨迹 | 组合律、误差传播 | LoopFormer、CTM、DPM-Solver |
| 停止和推理预算 | 当前状态稳定仍可能存在后续收益 | 终点可观测性、最优停止 | 自适应采样、内外层联合预算 |

RNN、数值分析与控制理论都有这些对象的早期形式；研究应明确现代 Transformer 的注意力记忆、预训练改造和部署预算究竟改变了什么。

## 2. 残差缩放：普通深度规则需要纳入跨轮相关性

令 $\Delta h=\eta\sum_{r=1}^K u_r$，假设 $\mathbb E\|u_r\|^2=v$ 且不同轮 $\mathbb E\langle u_r,u_s\rangle=\rho v$，展开得
$$\mathbb E\|\Delta h\|^2=\eta^2v[K+\rho K(K-1)].$$
弱相关对应 $\eta\sim K^{-1/2}$，正相关保持常数量级对应 $\eta\sim K^{-1}$（对平均二阶矩的近似，增量统计随尺度改变时需实测）。实际应记录完整 Gram 矩阵 $G_{rs}=\mathbb E\langle u_r,u_s\rangle$，而不是预设 $\rho$ 常数。

Loop-$\mu$P [L6] 在其无限宽单层循环参数化下得到 $1/K$ 缩放与循环数无关的学习率，推广至 $L$ 个不同层重复 $K$ 次时提出 $1/(K\sqrt L)$，报告超参迁移实验。其结论不能无条件推广到任意预训练 MoE 或逐轮门控结构。

EDM [D7] 对噪声阶段进行输入/输出/skip/噪声条件的预条件：
$$D_\theta(x;\sigma)=c_{\rm skip}(\sigma)x+c_{\rm out}(\sigma)F_\theta(c_{\rm in}(\sigma)x;c_{\rm noise}(\sigma)).$$
可迁移原则：固定共享核心，阶段接口承担尺度适配；以实测统计和梯度为依据设计残差尺度、归一化、损失权重。外层采样步长与内层残差缩放独立。GLM 的两轮门控是相关工业实例。

## 3. 完整执行敏感度：训练梯度和参数误差的共同算子

固定条件、随机数、阶段计划，在可微轨迹上定义 $A_r=D_ZT_{\theta,r}$、$B_r=D_\theta T_{\theta,r}$、$C=D_Z\mathcal R$、$\Phi_{j,i}=A_{j-1}\cdots A_i$（$\Phi_{i,i}=I$）。初始态与读出不含所讨论的共享参数时，链式法则给出
$$S_K=D_\theta y=\sum_{r=0}^{K-1}C\Phi_{K,r+1}B_r.$$
训练 $\nabla_\theta L=S_K^\top\nabla_yL$；共享参数近似 $\delta y=S_K\delta\theta+o(\|\delta\theta\|)$（二阶光滑有 $O(\|\delta\theta\|^2)$）。这是同一个执行 Jacobian 的两个使用方向。若初始态或读出含参数，补上直接路径。top-k、MoE 路由和停止切换需要真实执行干预来补充局部分析。

令 $v_r=C\Phi_{K,r+1}B_r\delta\theta$：
$$\|\delta y\|^2\simeq\sum_r\|v_r\|^2+2\sum_{r<s}\langle v_r,v_s\rangle.$$
单轮重构漏掉了共享偏差的相干积累/抵消。例 $h_{r+1}=h_r+\theta$，固定偏差引起 $K\delta\theta$，若一次采样的零均值偏差方差为 $\sigma^2$，则终点方差 $K^2\sigma^2$；若每轮独立抽样，则为 $K\sigma^2$。真实增长由传播方向决定。

输出度量 $W\succeq0$ 下的局部参数几何：
$$G_K=S_K^\top WS_K,\quad \|\delta y\|_W^2\simeq\delta\theta^\top G_K\delta\theta.$$
LM logits 的局部 KL 取 $W=\operatorname{diag}(p)-pp^\top$（KL 有 $1/2$ 系数）；平方输出距离或广义 Gauss–Newton 使用各自的 $W$。这并不等同于任务正确率的因果提升。状态扰动对应 $P_r=\Phi_{K,r}^\top C^\top W C\Phi_{K,r}$。JVP/VJP、随机投影和实际 rollout 可估计方向性敏感度，无需显式巨矩阵。

### 3.1 压缩迁移与新近查重

Q-Diffusion [D8] 已对跨噪声时间统计进行校准。Loop 可共享权重、按轮分别激活校准，并对实际预算分布优化
$$\min_{\tilde\theta}\mathbb E_{c,K}D(y_{\theta,K}(c),y_{\tilde\theta,K}(c)).$$
缓存精度同样可按未来传播后的敏感度分配；普通单轮 MSE 是辅助量。DEQ Jacobian regularization [L7] 提供低成本 Jacobian 估计，但有限步任务应优先关注实际扰动方向的终点作用。

2026-11-24 补充发现 LoopPrune [X7]（11 月 17 日预印本）：直接面向 looped diffusion 的结构剪枝，结合共享移除影响与位置变化做重要性估计、汇聚跨轮输入统计重构、在相同 student state 上匹配完整去噪预测与文本引导。作者报告 Looped-DiT B/16 的 30% 结构剪枝保留 93.14% 的六项基准平均表现（作者指标），以及其他架构上的显存/延迟收益。需要阅读全文与代码再复现。它与跨轮联合校准高度重叠，因此不能将“考虑所有循环的剪枝/重构”本身作为新颖性。仍有区分力的切入：显式分解共享扰动的跨轮终点交叉项、验证方向预测力、在不同循环及采样预算下联合分配精度。

## 4. 训练分布与阶段任务冲突

确定性更新的状态占据分布满足
$$\mu_{\theta,r+1}=(T_{\theta,r})_\#\mu_{\theta,r}.$$
循环参数、深度、缓存近似改变后续输入；相同核心面对多阶段分布。标准 diffusion 的 $q_t(x_t\mid x_0)$ 外部指定，可以独立采样训练状态；Loop 没有自然的正确隐态标签，因此任意高斯扰动不自动获得去噪的语义。

Analog Bits [D9] 将上一次数据估计作为自条件，Self Forcing [D10] 以自生成历史替换真实历史来训练自回归视频生成。可迁移的是让 Loop 接触真实自产状态、缓存近似和不同预算下的中间状态；局部修正目标应是任务读出、教师后续预测或明确中间任务。

多出口训练 $L=\sum_K w_KL_K$，$g_K=\nabla_\theta L_K$。梯度 Gram $g_K^\top g_{K'}$ 描述阶段/预算任务冲突；Min-SNR [D11] 从多任务角度处理 diffusion 时间权重。Loop 没有天然噪声 SNR，不应直接照搬其权重公式。先测梯度对齐、部署预算和任务收益，再选择权重或低成本阶段接口。单个终点损失中不同调用的 BPTT 项是链式法则，不应随意重加权而声称仍在优化原目标。

## 5. 记忆与未来信用：把缓存看作动态状态

线性示例：
$$h_{r+1}=Ah_r+Bm_r,\quad m_{r+1}=Ch_r+Dm_r,$$
消去 $m$ 得
$$h_{r+1}=Ah_r+BD^rm_0+\sum_{j=0}^{r-1}BD^{r-1-j}Ch_j.$$
显式记忆消去后出现历史核。压缩 KV、蒸馏多轮状态必须检查后续任务相关闭合，而非只对齐当前读出。

设第一轮写 $M_1=\operatorname{Write}_\theta(H_1)$，后续损失依赖它。未来梯度包含
$$(D_\theta\operatorname{Write})^\top\nabla_{M_1}L_{\rm future}$$
和经 $H_1$ 的链路。用自产记忆改善前向分布，与是否让未来损失训练过去写入，是两个不同设计轴。

| 方法（自回归视频） | 自产历史 | 未来损失到历史 KV | 完整历史采样反传 |
|---|---|---|---|
| Self Forcing 相关方案 | 是 | 截断 | 否 |
| Self Gradient Forcing (SGF) [D12] | 是 | 重构历史 KV 恢复 | 否，历史 latent detach |
| dKV Forcing [D13] | 是 | 持久可微缓存 | 否，指定 detach 边界 |
| 完整可微 BPTT | 是 | 是 | 是 |

SGF 两阶段：无梯度 rollout 记录自生成历史、采样退出步及当前 noisy latent；并行重构相应因果图，使未来损失训练历史 KV。dKV 用 persistent differentiable pages 维护缓存，避免 checkpoint replay 重复写入，在指定边界内计算梯度；发现生成与未来记忆写入梯度负对齐，增加只在 write 启用的低秩残差。二者原作者证据属于视频生成，不是 Loop LM 的实验证明。

可迁移至 GLM 类两轮结构：使用真实前向轨迹，选择性恢复未来读到过去写的梯度；比较共享写入和阶段专用低秩写入。控制前向完全一致，只改变反传边，区分状态分布收益和记忆信用收益。GLM 第一轮 gate 小不代表 KV 贡献小；同时分析主状态和 KV 的传播 Jacobian（见 [02](02_glm_and_learning_mechanisms.md)）。GLM 实际是否 detach 未公开。

## 6. 可变深度：延长计算与细化轨迹应分别处理

普通 $F^K$ 增加轮数改变复合算子，可能产生新算法步骤；固定终点的连续 ODE 增加网格点主要改善离散精度；蒸馏将多个教师调用压缩成少量学生调用。反例 $F(h)=-h$ 是单位 Euler 的 $v(h)=-2h$，但 $F^2(h)=h$，而对应 ODE 在 2 个时间单位后为 $e^{-4}h$。将任意离散 Loop 重写成 Euler 再换高阶 solver 可能改变任务。

DPM-Solver [D14] 利用已定义 diffusion ODE 的半线性结构。LoopFormer [L3] 学习时间/步长条件与终点一致性。CTM [D6] 学习区间映射，并比较共同干净终点。可直接训练
$$S_\phi(Z,a,b)\approx T_{b\leftarrow a}(Z),\qquad S_{c\leftarrow a}\approx S_{c\leftarrow b}\circ S_{b\leftarrow a}.$$
原离散教师区间组合律精确；学生仅近似。未见区间分割是关键测试。完整状态和记忆必须定义。比较两个中间态时，令
$$Q_r(Z)=\mathcal R(T_{K\leftarrow r}(Z)),\qquad D(Q_r(Z),Q_r(\tilde Z))$$
作为共同后缀的终点度量，连接非线性真实 rollout 与局部 $C\Phi$。

提前退出是成本与收益决策：估计继续运行的条件期望任务改善，权衡新增延迟/显存/计算。当前隐态变化小、logits 不变或 gate 小不能单独证明无需后续计算。实际省时要求真正跳过/缩小计算，并计入调度开销。

## 7. 联合预算与下一步

内部循环与外部采样是两条轴。若局部近似 $E(N,K)=aN^{-p}+bK^{-q}$，成本正比 $NK$，内部最优点满足 $paN^{-p}=qbK^{-q}$。幂律只是可拟合的简化，误差可能饱和、非单调、相互影响；真实需计入 pre/post 成本、阶段传播和质量。LiFT [X2] 为联合预算提供直接先例。进一步可以按样本、生成阶段和预测可信度分配循环预算。

三条主线分别归入 [04](04_research_questions.md)：跨轮相关结构与压缩敏感度、面向未来的记忆信用、可组合与预算可调的计算。