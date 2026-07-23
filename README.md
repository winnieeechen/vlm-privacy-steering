# VLM Privacy Steering: Dual-Adaptive Attentive Activation Steering

Method 14 is a self-contained activation-steering method for controlling the
location-disclosure granularity of free-form VLM responses. Label A means
refusal, B means broad location, and C means exact location.

The method routes from the image-and-prompt hidden state, retrieves a
query-specific behavior direction, and injects that direction while the VLM
generates its answer. It does not inspect a generated response to infer a
source label, use an LLM to choose the steering target, alter the prompt, or
select among multiple generated answers.

```text
image + prompt -> L29 condition router -> target/confidence
               -> attentive L28 behavior vector -> steered response
```

## Method

Method 14 将 **where to steer** 和 **how to steer** 分开处理。Offline 阶段先构建
两个 non-parametric memory；inference 时，Condition Memory 负责判断目标披露等级
A/B/C，Behavior Memory 则负责为当前 query 构造实际使用的 steering vector。

### 1. Offline：构建两个 Memory

#### Condition Memory：用于 Target Routing

对于每个 training image + prompt，不提供 answer，只对 VLM 做一次 forward，并
提取 Layer 29 的 prompt-last hidden state：

$$
c_i=h^{\mathrm{prompt}}_{29,i}[-1].
$$

首先计算所有 training condition states 的中心 $\mu_{\mathrm{route}}$，然后对每个
state 做 centering 和 L2 normalization：

$$
r_i=\operatorname{unit}(c_i-\mu_{\mathrm{route}}).
$$

Condition Memory 保存：

$$
\mathcal M_{\mathrm{route}}=\{(r_i,y_i^{\mathrm{true}})\}_{i=1}^{N},
$$

其中 $r_i$ 是用于 retrieval 的 condition feature，
$y_i^{\mathrm{true}}\in\{A,B,C\}$ 是 benchmark 提供的 human target label。
这个 memory 不保存回答内容，只描述“图片与 prompt 的 hidden representation”以及
“该样本应该对应的披露等级”。后续 kNN router 将在这里检索和投票。实验中一共保存
717 条 routing entries。

#### Behavior Memory：用于构造 Query-Specific Steering Vector

对同一批 training examples，使用 teacher forcing 输入完整的 base natural-language
response，并提取 Layer 28 的所有 answer-token hidden states。对 answer token 取均值：

$$
a_i=\frac{1}{T_i}\sum_{t=1}^{T_i}h^{\mathrm{answer}}_{28,i,t}.
$$

GPT-4o-mini 对完整 response 判断披露等级
$y_i^{\mathrm{resp}}\in\{A,B,C,D\}$，其中 D 会被排除。将 answer representation
$a_i$ 与同一个样本的 condition feature $r_i$ 配对，构成 Behavior Memory：

$$
\mathcal M_{\mathrm{behavior}}
=\{(r_i,a_i,y_i^{\mathrm{resp}})\}.
$$

它可以看作一个 key-value memory：

- $r_i$ 是 condition-space **key**，用于判断哪些训练样本与当前 query 相似；
- $a_i$ 是 answer-space **value**，用于构造 activation steering vector；
- $y_i^{\mathrm{resp}}$ 是 response 的 behavior label，用于区分 A/B/C memory。

实验中 Behavior Memory 包含 710 条 response：245 A、387 B、78 C。两个 memory
通过相同的 condition feature $r_i$ 连接，但作用不同：Condition Memory 决定
**往哪里 steer**，Behavior Memory 决定 **使用什么 vector steer**。

### 2. Inference：使用 Condition Memory 进行 Target Routing

对于 test image + prompt $x$，提取 Layer 29 prompt-last hidden state，并使用
training 阶段相同的 center 做 normalization：

$$
r(x)=\operatorname{unit}(c(x)-\mu_{\mathrm{route}}).
$$

计算 $r(x)$ 与 Condition Memory 中所有 $r_i$ 的 cosine similarity，取 top-$k$
neighbors（$k=11$）。邻居中 A/B/C true label 所占比例即 target probabilities：

$$
p_y(x)=\frac{1}{k}\sum_{i\in\mathcal N_k(x)}
\mathbf 1[y_i^{\mathrm{true}}=y],\qquad y\in\{A,B,C\}.
$$

raw target 为 $t_0=\arg\max_y p_y(x)$，routing confidence 为
$q(x)=\max_y p_y(x)$。随后使用 confidence gate 得到最终 target：

$$
t(x)=
\begin{cases}
A,&t_0=A\ \text{and}\ q(x)\ge 0.45,\\
C,&t_0=C\ \text{and}\ q(x)\ge 0.55,\\
B,&\text{otherwise}.
\end{cases}
$$

因此，低置信度的 A/C prediction 不会被强行 intervention，而是回退到 B。target B
表示保留模型原始生成过程，不进行 activation steering。

### 3. 使用 Behavior Memory 构造 Local Vector

得到 target 后，使用同一个 query condition feature $r(x)$ 在 Behavior Memory
中检索。若 target=A，分别从 response label 为 A 和 B 的子集中取 top-$m$
neighbors；若 target=C，则分别从 C 和 B 中检索。这里 $m=24$。

检索仍使用 condition key 之间的 cosine similarity，但最终读取的是对应的 answer
representation $a_i$。使用 temperature $\tau=0.05$ 的 softmax weights 计算
similarity-weighted local centroid：

$$
w_i^y(x)=\frac{\exp(\operatorname{sim}(r(x),r_i)/\tau)}
{\sum_{j\in\mathcal N_m^y(x)}
\exp(\operatorname{sim}(r(x),r_j)/\tau)},
\qquad
\mu_y^{\mathrm{local}}(x)=\sum_{i\in\mathcal N_m^y(x)}w_i^y(x)a_i.
$$

由此得到当前 query 对应的 local behavior vectors：

$$
v_A^{\mathrm{local}}(x)=\mu_A^{\mathrm{local}}(x)-
\mu_B^{\mathrm{local}}(x),
$$

$$
v_C^{\mathrm{local}}(x)=\mu_C^{\mathrm{local}}(x)-
\mu_B^{\mathrm{local}}(x).
$$

target A 使用 A−B，表示从 broad disclosure 朝 refusal 方向移动；target C 使用
C−B，表示从 broad disclosure 朝 exact disclosure 方向移动。只构造当前 target
需要的 local vector；target B 跳过该步骤并使用 zero vector。

### 4. 融合 Local Vector 与 Global Vector

Local vector 来自与当前 query 最相似的少量样本，适应性更强，但容易受到 neighbor
noise 影响。Global vector 则使用 Behavior Memory 中某一 response class 的全部
answer representations 计算，更稳定：

$$
\mu_y^{\mathrm{global}}=\frac{1}{|\mathcal I_y|}
\sum_{i\in\mathcal I_y}a_i,
$$

$$
v_A^{\mathrm{global}}=\mu_A^{\mathrm{global}}-
\mu_B^{\mathrm{global}},\qquad
v_C^{\mathrm{global}}=\mu_C^{\mathrm{global}}-
\mu_B^{\mathrm{global}}.
$$

对于 target $y\in\{A,C\}$，先计算 local/global vector 的 cosine similarity。
如果方向相反（$\rho<0$），先翻转 local vector；然后将 local vector 的 norm 调整为
与 global vector 相同：

$$
\rho=\cos(v_y^{\mathrm{local}},v_y^{\mathrm{global}}),\qquad
\rho<0\Rightarrow v_y^{\mathrm{local}}\leftarrow-v_y^{\mathrm{local}},
$$

$$
\bar v_y^{\mathrm{local}}=
v_y^{\mathrm{local}}\frac{\lVert v_y^{\mathrm{global}}\rVert_2}
{\lVert v_y^{\mathrm{local}}\rVert_2}.
$$

融合时，local weight 最大为 $\lambda_{\max}=0.25$，并根据 local/global agreement
自动缩放：

$$
\lambda=0.25|\rho|,\qquad
\tilde v_y=(1-\lambda)v_y^{\mathrm{global}}
+\lambda\bar v_y^{\mathrm{local}}.
$$

最后将 $\tilde v_y$ 再次 rescale 到 global-vector norm，得到最终 fused vector
$v_y(x)$。因此 global vector 始终提供稳定的主方向，local vector 只根据当前图片
进行最多 25% 的 query-specific adjustment；当两者 agreement 较低时，local
contribution 会自动减小。

### 5. Dual-Adaptive Activation Injection

除了 vector direction 会随 query 改变，steering magnitude 也会根据 router
confidence 自适应变化。对于 $y\in\{A,C\}$：

$$
s_y(x)=\operatorname{clip}\left(
\frac{q(x)-\theta_y}{1-\theta_y},0,1\right),\qquad
\alpha_y(x)=\alpha_y^{\min}+s_y(x)
(\alpha_y^{\max}-\alpha_y^{\min}).
$$

最终选择的 schedules 为：A 使用 $\theta_A=0.45$、
$\alpha_A\in[1.0,1.5]$；C 使用 $\theta_C=0.55$、
$\alpha_C\in[3.0,4.0]$。在每个 decoding step，将 intervention 注入 Layer 28
的 last-token hidden state：

$$
h'_{28,t}=h_{28,t}+\Delta h(x),\qquad
\Delta h(x)=
\begin{cases}
\alpha_A(x)v_A(x),&t(x)=A,\\
0,&t(x)=B,\\
\alpha_C(x)v_C(x),&t(x)=C.
\end{cases}
$$

Inference 时，target routing 和 local-vector retrieval 都只依赖当前 VLM hidden
state 与两个 offline memory。GPT-4o-mini 只在 offline 阶段为 training responses
提供 behavior labels，并在生成结束后用于 evaluation；它不会在 inference 时判断
source，也不会向 VLM 提供 steering target。

## Files

```text
scripts/14_free/
  method14_common.py                    # Qwen, judge, CSV, and evaluation helpers
  00_prepare_assets.py                  # extract train states and build all assets
  01_run_dual_adaptive_steering.py      # run val/test activation steering
  02_sweep_confidence_val.sh            # optional validation gate ablation
  03_summarize_confidence_val.py        # rank validation gate candidates
  04_summarize_results.py               # audit a completed result CSV
  05_export_public_results.py            # export portable CSVs for GitHub
```

Method 14 does not import code from any earlier steering method. Generated
assets are stored under `outputs/14_free/assets` by default.


## Results

All labels below use Duke GPT-4o-mini. D outputs are retained in the denominator
and counted as incorrect.

| Split | Base | Method 14 | Recall A | Recall B | Recall C |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 102/238 (42.9%) | **148/238 (62.2%)** | 72.2% | 42.6% | 61.7% |
| Test | 98/243 (40.3%) | **140/243 (57.6%)** | 61.2% | 46.9% | 59.4% |

Method 14 improves validation accuracy by `19.3` percentage points and test
accuracy by `17.3` percentage points over the stored free-form base responses.

但这里val和test也挺奇怪的，感觉val比test好很多啊，不知道是不是一开始split的问题。

Complete per-example artifacts:

- [Validation results](results/14_free/val_results.csv)
- [Test results](results/14_free/test_results.csv)
- [Validation confidence sweep](results/14_free/validation_confidence_sweep.csv)

### Confidence-gate ablation

The selected `A=0.45, C=0.55` thresholds had the best validation accuracy and
macro recall:

| A gate | C gate | Correct | Recall A/B/C | Macro recall |
| ---: | ---: | ---: | ---: | ---: |
| **0.45** | **0.55** | **148/238 (62.2%)** | 72.2/42.6/61.7% | **58.8%** |
| 0.50 | 0.55 | 144/238 (60.5%) | 67.0/44.7/61.7% | 57.8% |
| 0.55 | 0.65 | 141/238 (59.2%) | 70.1/48.9/53.2% | 57.4% |
| 0.45 | 0.65 | 141/238 (59.2%) | 73.2/42.6/53.2% | 56.3% |
| 0.50 | 0.65 | 136/238 (57.1%) | 68.0/44.7/52.1% | 54.9% |

## Evaluation note

The stored base responses were sampled at temperature `0.7`, while the reported
Method 14 runs use greedy decoding (`temperature=0`). The table measures the
complete selected system against those stored base responses; a strictly
matched causal ablation should additionally regenerate a zero-vector greedy
baseline.
