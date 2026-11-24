# 循环计算与扩散生成：共同数学结构

2026-11-24 · 数学基础。文献 ID 对应 [参考索引](../references/README.md)。

## 1. 研究对象与抽象层级

Looped Transformer 沿计算深度复用模块。设输入条件 $c$、固定表征 $E=P(c)$、隐藏状态 $H_r\in\mathbb R^{B\times N\times d}$，可写作
$$H_{r+1}=F_\theta(H_r,E,\tau_r),\quad y=R(H_K).$$
Universal Transformer 使用循环步编码；Huginn 随机初始化隐状态并重复注入输入，不显式给共享核心传入循环编号 [L1,L2]。GLM-5.4 有阶段专用门控和跨轮 KV。

Diffusion 采用噪声增加的时间约定：
$$X_t=\alpha_t X_0+\sigma_t\epsilon,\quad \epsilon\sim\mathcal N(0,I).$$
生成时预测噪声、score、干净样本或速度，并由求解器推进 $X_{t_{k+1}}=U_{\theta,k}(X_{t_k},c,\zeta_k)$。通常 $t_{k+1}<t_k$。Flow matching 学习概率路径速度场；概率流 ODE 与对应 SDE 可以具有相同单时刻边缘而非相同路径分布 [D1–D4]。

共同完整状态为 $Z=(H,M)$ 或 $(X,M)$；$M$ 包含仍影响后续转移的 KV、历史预测、索引等。
$$Z_{r+1}=T_\theta(Z_r;c,u_r,\xi_r),\qquad y=\mathcal R(Z_K).$$
$u_r$ 是阶段、步长、门控、精度或缓存等控制。权重共享是计算核心的约束，完整转移可随阶段而变。Loop 是计算组织方式，diffusion 是生成概率路径与更新的组织方式；可独立组合。

对两个系统 $F_r,U_r$，应根据研究目的选择需要保持的关系：

| 对应 | 数学条件 | 可迁移对象 |
|---|---|---|
| 动力学共轭 | $\pi F_r=U_r\pi$，且 $\pi$ 可逆 | 逐步更新、固定点；度量稳定性另需正则性 |
| 半共轭 | 同式，$\pi$ 非可逆 | 闭合的任务状态；需保证同一纤维下一步读出一致 |
| 有限步终点 | $R_H(H_K)=R_X(X_N)$ | 区间蒸馏、跳步、任务读出 |
| 条件分布 | $\operatorname{Law}(R_H(H_K)\mid c)=\operatorname{Law}(R_X(X_N)\mid c)$ | 生成质量、输运路径自由度 |

共轭整个采样过程还需要匹配初始化分布 $\pi_\#\mu_H=\mu_X$ 和读出。以下是有具体条件的局部对应，并非宣称任意 Loop 和 diffusion 全局同构。

## 2. 精确代数对应：DDIM 是特定坐标中的门控循环

确定性 DDIM（eta=0）采用干净样本预测 $D_\theta(x_t,t,c)$：
$$X_s=\alpha_sD_\theta(X_t,t,c)+\sigma_s\frac{X_t-\alpha_tD_\theta(X_t,t,c)}{\sigma_t}.$$
令 $Y_t=X_t/\alpha_t,\ \lambda_t=\sigma_t/\alpha_t,\ \widetilde D_\theta(y,t,c)=D_\theta(\alpha_t y,t,c)$。在 $\alpha_s,\alpha_t>0,\sigma_t>0$ 的区间：
$$Y_s=\rho_{s,t}Y_t+(1-\rho_{s,t})\widetilde D_\theta(Y_t,t,c),\qquad \rho_{s,t}=\lambda_s/\lambda_t.$$
当噪声信号比随生成方向下降时 $0\le\rho\le1$。这正是 $h_{r+1}=h_r+g_r(F_\theta(h_r)-h_r)$，其中 $g_r=1-\rho$。对任意预测器是恒等式；原始网络参数化与生成质量仍由训练和求解器决定 [D2]。$\alpha=0$ 的边界须另处理。

对高斯观测 $Y=X_0+\lambda\epsilon$，若 $D_\lambda(y,c)=\mathbb E[X_0\mid Y=y,c]$ 为精确后验均值，微分高斯似然可得
$$D_\lambda-y=\lambda^2\nabla_y\log p_\lambda(y\mid c),\quad JD_\lambda=\lambda^{-2}\operatorname{Cov}(X_0\mid y,c).$$
因此 $J G=\rho I+(1-\rho)JD_\lambda$，其增益由后验协方差和门控决定。这个对称半正定结构属于精确后验均值，不自动适用于任意学习到的循环算子。

可算反例/启发：$X_0\in\{-a,a\}$ 等概率，
$$D_\lambda(y)=a\tanh(ay/\lambda^2),\quad D'_\lambda(y)=(a^2/\lambda^2)\operatorname{sech}^2(ay/\lambda^2).$$
取 $a=2,\lambda=1,\rho=0.5$，$G'(0)=2.5$，$G'(2)\approx0.502682$。分支附近扩张、分支内收缩。由此提出 Loop 假说：任务相关的分支方向需要辨别能力，同分支的无关扰动可以压缩。不能把此例直接当成真实语言模型的机制证据。

## 3. 内部计算深度与生成时间的独立性

定义 $H^{(k,r)}_{bifa}$，可将 token 轴展开为视频帧 $i$、空间位置 $f$，$a$ 为通道。采样步 $k$、循环深度 $r$ 和视频帧轴不同。
$$H^{(k,0)}=P(X_{t_k},t_k,c),\quad H^{(k,r+1)}=F_\theta(H^{(k,r)},X_{t_k},t_k,c),$$
$$\widehat v_k=R(H^{(k,R_k)}),\qquad X_{t_{k+1}}=\operatorname{SolverStep}(X_{t_k},\widehat v_k,t_k,t_{k+1}).$$
内层改善当前位置的预测，外层搬运样本分布。

- Looped-DiT [X1] 共享中间块，每轮读出预测同一干净图像；通过深监督和 self-modulating attention 控制局部信息覆盖。原作者观察到特定预算下增加内轮比增加采样步更有效；规模、分辨率、训练开销均有边界。
- LiFT [X2] 在固定观测 $z=(X_t,t,c)$ 下，以停止梯度的初始速度估计 $b$ 为锚，连续深度 $\tau$ 的回归目标 $\bar v_\tau=(1-\tau)b+\tau v^\star_{\rm sample}$。平方损失最优条件均值为 $(1-\tau)b+\tau\mathbb E[v^\star_{\rm sample}\mid z]$（$b$ 为给定 $z$ 的确定估计）。这规定了内部预测校正路径，并不要求循环本身是 ODE solver。

局部误差模型：
$$e_{k+1}\simeq A_ke_k+\Delta t_k\eta_{k,R_k}+d_k,$$
$$e_N\simeq\sum_k\Phi_{N,k+1}(\Delta t_k\eta_{k,R_k}+d_k),\qquad \Phi_{N,k+1}=A_{N-1}\cdots A_{k+1}.$$
$\eta$ 为预测误差，$d$ 为局部离散误差，符号由推进方向定义。一般求解器需使用实际 stage 权重与增广历史状态。成本 $C=\sum_k(C_{\rm pre}+R_kC_{\rm core}+C_{\rm post})$，真实延迟还依赖 KV、访存和并行度。内外轴预算应依据终点敏感度和实际边际收益。

## 4. 终点观测量、区间组合与一致性

有限离散轨迹的终点算子
$$Q_r(z)=\mathcal R(F_{K-1}\circ\cdots\circ F_r(z)),\quad Q_r(z)=Q_{r+1}(F_r(z)),\quad Q_K=\mathcal R.$$
确定性生成流的相应量 $Q(x,t)=\mathcal R(\Phi_{0\leftarrow t}(x))$，沿轨迹保持不变。在可微条件下满足 $\partial_t Q+DQ\,v_t=0$。Consistency Models [D5] 学习这种终点一致性；随机更新对应 $Q_r(z)=\mathbb E[Q_{r+1}(Z_{r+1})\mid Z_r=z]$。

可训练终点读出 $Q_\phi$，对齐相邻轮的教师终点预测并以正确答案或教师终点固定边界。目的不是冻结隐藏态，而是让答案预测可靠、可提前读出。

区间映射 $T_{b\leftarrow a}=F_{b-1}\circ\cdots\circ F_a$ 满足组合律 $T_{c\leftarrow a}=T_{c\leftarrow b}T_{b\leftarrow a}$。LoopFormer [L3] 已以阶段和步长调制及长短路径一致性实现弹性深度；CTM [D6] 给出区间映射和共同终点比较的 diffusion 经验。

## 5. 轨迹整体提升：联合固定点

给定初始噪声 $\xi$，$Z_0=\xi,\ Z_{k+1}=U_k(Z_k)$。定义 $\mathcal Z=(Z_1,\ldots,Z_N)$，
$$\mathcal G_\xi(\mathcal Z)=(U_0(\xi),U_1(Z_1),\ldots,U_{N-1}(Z_{N-1})).$$
原轨迹与 $\mathcal Z^\star=\mathcal G_\xi(\mathcal Z^\star)$ 一一对应；随机链需固定随机增量。DEQ-DDIM [X3] 以联合固定点进行并行求解和反演。同步 Jacobi 更新因链的三角结构可在至多 N 轮传播初始条件，但每轮成本和存储上升；加速收益来自并行/求解器而非恒等式本身。
GET [X4] 则通过噪声注入训练新的平衡生成器 $h^\star=F_\theta(h^\star,\xi,c)$，蒸馏噪声到样本的整体映射。对固定随机种子收敛，与跨种子多样性可以并存。

## 6. 状态充分性：找到当前不可见但未来有用的计算

若 $\pi(h)=\pi(h')$ 但 $\pi(F(h))\ne\pi(F(h'))$，读出状态不闭合。例如 $F(x,m)=(x+m,m/2)$、$\pi(x,m)=x$，状态 $(0,1)$ 和 $(0,-1)$ 当前读出相同，下一步分别为 1 和 -1。

平方可积条件下，
$$\min_G\mathbb E\|\pi(F(H))-G(\pi(H))\|^2=\mathbb E\operatorname{tr}\operatorname{Cov}(\pi(F(H))\mid\pi(H)).$$
对局部方向 $V_{jb}$，若 $D\pi(H)V=0$，但
$$[D(\pi\circ F)(H)V]_\mu=\sum_{ia,jb}\frac{\partial\pi_\mu(F(H))}{\partial F_{ia}}\frac{\partial F_{ia}}{\partial H_{jb}}V_{jb}\ne0,$$
则此方向当前读出不可见、后续会影响读出。用 JVP/VJP 测量，避免物化 Jacobian。可将有用部分作为持久记忆 $M_k$ 加入 diffusion 状态，但需要在真实 rollout 上验证与成本比较。

若只保持边缘概率路径，连续性方程 $\partial_t p_t+\nabla\cdot(p_t v_t)=0$ 允许加入满足 $\nabla\cdot(p_tw_t)=0$ 的场（在良定性与边界条件下）。对各向同性高斯，$w_t(x)=\Omega_t x,\Omega_t^\top=-\Omega_t$ 即为例子。轨迹可变、边缘保持，提示研究内部运动的自由度。

## 7. 已有迁移与证据范围

- Geiping 等 [X5] 将 diffusion forcing 的异步精炼调度用于 recurrent-depth LM，边解码新 token 边继续精炼早期 token 的隐状态；在现有 3.5B 模型上报告最高约 5 倍加速。依赖可解码中间态和并行计算接口。
- Svete 与 Sabharwal [X6] 在有限精度、对数宽度、额外 padding 工作空间及特定更新/随机源设定下，证明 masked diffusion 与 polynomially padded looped Transformers 的计算表达对应；不是一般连续图像 diffusion 与任意 LM 的逐状态等价。
- 下一步在 [GLM 与训练机制](02_glm_and_learning_mechanisms.md) 中恢复实际计算图，在 [共同问题](03_shared_problems_and_transfer.md) 中从上述数学对象推导训练和压缩方法。