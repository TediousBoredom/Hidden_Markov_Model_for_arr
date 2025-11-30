


请寻找⼀个合适的数据集，利⽤ EM 算法拟合⼀个合适的 Hidden Markov Model，展示并讨
论在测试集上的表现。要求：
1. 给出公开数据集的⼀般性描述与链接；
2. 给出所建⽴的 Hidden Markov Model 具体形式与需训练的模型参数；
3. 给出实现算法的伪代码；
4. 展示并讨论其在不少于⼀种 Inference 问题（包含 Filtering，Smoothing，Prediction，
MAP estimation）上的表现；
5. 正⽂篇幅控制在 2 ⻚ A4 纸以内； 关键代码部分可作为附件提交。 提交截⽌时间为 12 ⽉
11 ⽇晚上 18:30 上课前。注意，若发现明显抄袭、作假等学术不端⾏为，成绩减半。解释


1．原理对应（EM ↔ Baum－Welch）
- 通用 EM：
- 完整数据对数似然 $\ell_c(\theta)=\sum_i \log p\left(x_i, z_i \mid \theta\right)$ 。
- E－step：计算 $Q\left(\theta, \theta^{(t-1)}\right)=\mathbb{E}_{z \sim p\left(z \mid x, \theta^{(t-1)}\right)}\left[\ell_c(\theta)\right]$（即对隐变量取后验期望）。
- M－step：$\theta^{(t)}=\arg \max _\theta Q\left(\theta, \theta^{(t-1)}\right)$ 。
－HMM（Baum－Welch）：
- 数据项是整条观测序列 $y_{1: T}$ ，隐变量是整个状态序列 $z_{1: T}$ 。
- E－step：用 forward－backward 算法计算每个时间点的后验边缘 $\gamma_t(k)=P\left(z_t=k \mid y_{1: T}\right)$（类似 GMM 的 $\left.r_{i k}\right)$ ，以及相邻时刻的联合后验 $\xi_t(i, j)=P\left(z_t=i, z_{t+1}=j \mid y_{1: T}\right)$（用于更新转移）。
- M－step：用这些期望统计量更新参数：
- 初始分布 $\pi_k \leftarrow \gamma_1(k)$（或归一化的累积）
- 转移 $A_{i j} \leftarrow \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$
- 发射（若 Gaussian emission）$: ~ \mu_k \leftarrow \frac{\sum_t \gamma_t(k) y_t}{\sum_t \gamma_t(k)}, ~ \Sigma_k \leftarrow$
$$
\frac{\sum_t \gamma_t(k)\left(y_t-\mu_k\right)\left(y_t-\mu_k\right)^{\top}}{\sum_t \gamma_t(k)} .
$$

所以：GMM 的 $r_{i k} \leftrightarrow \mathrm{HMM}$ 的 $\gamma_t(k)$, GMM 的簇间共现 ↔ HMM 的 $\xi_t(i, j)$ 。理论保证（Jensen 不等式、 KL分解）也完全相同：每次 EM／Baum－Welch 都不减小观测数据对数似然。

2．HMM 具体形式（建议用于作业）
- 离散时间 HMM ，隐状态数 $K$（你可用 $\mathrm{K}=6$ 对应 HAR 的 6 类活动，或用无监督 K 选通过 BIC／AIC）。
- 发射分布：多元高斯（若用原始传感器流或 PCA 后的连续特征）；或 GMM per－state（更灵活）。
- 待估参数：$\theta=\left\{\pi, A, \mu_k, \Sigma_k\right\}_{k=1}^K$ 。


实现细节提示：

用 log-space 做 forward/backward 防止数值下溢；

对协方差加 ϵI 防止奇异；

如果观测维度高，可先做 PCA 降维到 5–20 维再建模。



4．Inference（Filtering／Smoothing／Prediction／MAP）与如何评测
－Filtering：计算 $P\left(z_t \mid y_{1: t}\right)$（在线后验）。用 $\alpha$ 递推并归一化。可用 $\arg \max _k P\left(z_t=k \mid y_{1: t}\right)$ 得到时刻标签 → 与真实标签比较计算逐时准确率。
- Smoothing：计算 $P\left(z_t \mid y_{1: T}\right) \propto \alpha_t \beta_t$ 。取 argmax 作逐时预测。通常 smoothing $\geq$ filtering。
- Prediction（一步或多步）：一步预测 $P\left(z_{t+1} \mid y_{1: t}\right)=P\left(z_t \mid y_{1: t}\right) A$ 。把预测的隐状态（或观测分布）与真实下一时刻比较，计算预测准确率。
－MAP（Viterbi）：整序列最可能路径 $\arg \max _{z_{1: T}} P\left(z_{1: T} \mid y_{1: T}\right)$ 用 Viterbi 算法，比较路径与真路径的逐时准确率或序列一致率。

评测建议：
－如果你有带标签的数据（例如 UCI HAR）且想把隐状态与真实类别对齐：用 Hungarian 算法把估计的状态编号重标号（permutation）再计算准确率／混淆矩阵。
－指标：逐时准确率（平均 over t 与序列）、类别级别 recall／precision、混淆矩阵；还可以报对数似然（训练／验证）和 BIC／AIC 做模型选择。

5．小结 \＆常见问题
－Baum－Welch 就是 EM－E－step 求出后验边缘 $\gamma_t$ 和 $\xi_t$ ，M－step 用这些期望更新 $\pi$ 、A、发射分布参数，直观上对应你讲的 GMM 推导（把 $r_{i k}$ 换成时间索引的 $\gamma$ ）。
－区别只是＂隐变量结构＂不同：GMM 的隐变量为每个数据点独立，HMM 的隐变量是有马尔可夫依赖的序列 →因此需要 forward／backward 来计算后验而不是简单地用 Bayes 一步计算 $r_{i k}$ 。
－对高维或复杂观测建议使用 GMM－HMM 或在发射上用更强的模型（神经网络条件发射）或先做降维。