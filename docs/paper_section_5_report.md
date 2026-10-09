# Paper Section 5: Imperfect State Information

This note tracks the implementation of the supplementary paper's imperfect-state
sequence. Section 5 uses one Actor and multiple Observers. The Actor does not
use the full state directly; instead, it selects one Observer, receives that
Observer's compressed state message, and acts from the message.

## Section 5.1: Learnable Observer Selection

Section 5.1 isolates Observer selection. The Observer compressions and Actor
policy are fixed, and the only learned parameter is the Actor's selection-logit
vector \(w \in \mathbb{R}^N\). Selection probabilities are

\[
\lambda_i(w) =
\frac{e^{w_i}}{\sum_j e^{w_j}}.
\]

At each step,

\[
k_t \sim \lambda(w), \qquad
m_t = f_{\phi_{k_t}}(s_t), \qquad
a_t \sim \pi(\cdot \mid m_t).
\]

The Section 5.1 loss is the REINFORCE loss for the sampled Observer index,

\[
\mathcal{L}_t = -r_t \log \lambda_{k_t}.
\]

For the selected Observer \(k_t\), the softmax-gradient update is

\[
w_{k_t,t+1}
\leftarrow
w_{k_t,t}
+
\alpha_w r_t(1-\lambda_{k_t}),
\]

and for each unselected Observer \(j \neq k_t\),

\[
w_{j,t+1}
\leftarrow
w_{j,t}
-
\alpha_w r_t\lambda_j.
\]

No Actor-policy or Observer-compression updates are performed in this stage.

## Controlled Test

The state is encoded by binary features

\[
s=(x_1,x_2)\in\{(0,0),(1,0),(0,1),(1,1)\}.
\]

The useful state summary is the known sum

\[
z(s)=x_1+x_2.
\]

The fixed Observers are:

| Observer | Message |
| --- | --- |
| `zero` | \(0\) |
| `first_feature` | \(x_1\) |
| `sum` | \(x_1+x_2\) |

The fixed Actor decoder maps scalar messages near \(0,1,2\) to actions
\(0,1,2\). Correct actions receive reward \(1.0\), and incorrect actions
receive reward \(-0.1\). The `sum` Observer is therefore uniquely useful.

## Result

The experiment uses 5 seeds, 5000 steps, and learning rates
\(\{0.001,0.01,0.05,0.1\}\). The effective ceiling is the expected reward of
the fixed `sum` Observer, \(0.971\).

| Learning rate | Final expected | Final sampled | Final action prob. | Final \(P(\text{sum})\) |
| --- | ---: | ---: | ---: | ---: |
| 0.001 | 0.722 | 0.699 | 0.748 | 0.612 |
| 0.01 | 0.954 | 0.936 | 0.958 | 0.974 |
| 0.05 | 0.968 | 0.954 | 0.971 | 0.995 |
| 0.1 | 0.969 | 0.954 | 0.972 | 0.998 |

The learned selection distribution concentrates on the uniquely useful
Observer for learning rates \(0.01\) and above. The high-reward action
probability rises with the selection probability because the fixed decoder now
receives the useful message more often. The \(0.001\) run moves in the right
direction but is too slow within 5000 steps.

Generated files:

```text
configs/paper_5_1_observer_selection.yaml
experiments/run_paper_5_1.py
docs/figures/paper5_1_observer_selection.pdf
docs/figures/paper5_1_observer_selection.png
docs/figures/paper5_1_observer_selection.svg
outputs/paper_5_1/observer_selection_curves.csv
outputs/paper_5_1/observer_selection_summary.csv
```

## Section 5.2: Learnable Observer Selection and Message Compression

Section 5.2 keeps the Actor policy fixed but allows selected Observers to learn
their message functions. The learned parameters are now:

- selection logits \(w\);
- selected Observer compression parameters \(\phi_{k_t}\).

The following-logit update is unchanged from Section 5.1. The selected
Observer also receives the chain-rule update

\[
\phi_{k_t,t+1}
\leftarrow
\phi_{k_t,t}
+
\alpha_\phi r_t
\left[
\nabla_m \log \pi(a_t \mid m)
\right]_{m=m_t}
\nabla_{\phi_{k_t}}f_{\phi_{k_t}}(s_t).
\]

Observers that were not selected do not update on that step.

## Controlled 5.2 Test

The same binary state features are used:

\[
s=(x_1,x_2)\in\{(0,0),(1,0),(0,1),(1,1)\},
\qquad
z(s)=x_1+x_2.
\]

The Actor decoder is fixed. The three Observers have learnable one-layer
message functions, but their input masks differ:

| Observer | Available input | Interpretation |
| --- | --- | --- |
| `zero_input` | \((0,0)\) | no state information |
| `first_feature` | \((x_1,0)\) | partial state information |
| `full_sum` | \((x_1,x_2)\) | can learn the sum compression |

This makes the expected result identifiable: `full_sum` should learn a message
aligned with \(x_1+x_2\), should become the highest-reward Observer, and should
receive most of the selection probability.

## 5.2 Result

The experiment uses 5 seeds, 20000 steps, fixed \(\alpha_w=0.005\), and a sweep
over \(\alpha_\phi\).

| \(\alpha_\phi\) | Final expected | Final sampled | Final action prob. | Final \(P(\text{full})\) | Final \(R^2\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 0.972 | 0.989 | 0.974 | 0.986 | 1.000 |
| 0.01 | 0.972 | 0.987 | 0.974 | 0.985 | 1.000 |
| 0.02 | 0.972 | 0.987 | 0.975 | 0.985 | 1.000 |
| 0.05 | 0.864 | 0.883 | 0.876 | 0.795 | 1.000 |

Moderate compression learning rates learn the intended compression, raise the
Actor's high-reward action probability, and select the capable Observer. The
largest tested \(\alpha_\phi\) still learns an \(x_1+x_2\)-aligned message, but
the coupled dynamics are less stable and the selection probability on
`full_sum` is lower.

Generated files:

```text
configs/paper_5_2_selection_compression.yaml
experiments/run_paper_5_2.py
docs/figures/paper5_2_selection_compression.pdf
docs/figures/paper5_2_selection_compression.png
docs/figures/paper5_2_selection_compression.svg
outputs/paper_5_2/selection_compression_curves.csv
outputs/paper_5_2/selection_compression_summary.csv
```

## Section 5.3: Learnable Observer Selection, Message Compression, and Actor Policy

Section 5.3 learns all components in the one-Actor/multiple-Observer
imperfect-state setting:

- the Actor policy parameters \(\theta\);
- the selected Observer compression parameters \(\phi_{k_t}\);
- the Observer-selection logits \(w\).

At each step,

\[
k_t \sim \lambda(w), \qquad
m_t = f_{\phi_{k_t}}(s_t), \qquad
a_t \sim \pi_\theta(\cdot \mid m_t).
\]

The Actor receives the standard policy-gradient update,

\[
\theta_{t+1}
\leftarrow
\theta_t
+
\alpha_\theta r_t \nabla_\theta \log \pi_\theta(a_t \mid m_t).
\]

The selected Observer receives the same chain-rule compression update as
Section 5.2,

\[
\phi_{k_t,t+1}
\leftarrow
\phi_{k_t,t}
+
\alpha_\phi r_t
\left[
\nabla_m \log \pi_\theta(a_t \mid m)
\right]_{m=m_t}
\nabla_{\phi_{k_t}}f_{\phi_{k_t}}(s_t).
\]

The following-logit update is unchanged from Section 5.1:

\[
w_{k_t,t+1}
\leftarrow
w_{k_t,t}
+
\alpha_w r_t(1-\lambda_{k_t}),
\qquad
w_{j,t+1}
\leftarrow
w_{j,t}
-
\alpha_w r_t\lambda_j
\quad (j\neq k_t).
\]

## Controlled 5.3 Test

The fully joint test uses a balanced three-state sum task:

\[
s\in\{(0,0),(1,0),(1,1)\},
\qquad
\text{correct action}=x_1+x_2\in\{0,1,2\}.
\]

This avoids the action-frequency bias of the four-state sum task used in
Sections 5.1 and 5.2, where action 1 is correct for two of the four states. In
the fully joint case, that bias can create a local solution where the Actor
overuses the middle action. The three-state task keeps the diagnostic balanced
while preserving the same sum-compression structure.

The Observers use learnable one-layer message functions with fixed input
masks:

| Observer | Available input | Interpretation |
| --- | --- | --- |
| `zero_input` | \((0,0)\) | no state information |
| `first_feature` | \((x_1,0)\) | partial state information |
| `full_state` | \((x_1,x_2)\) | can distinguish all target actions |

Because the Actor policy is learned, the expected final message is not unique.
The full-state Observer does not need to learn the literal message
\(m=x_1+x_2\). A reversed, shifted, or rescaled one-dimensional code is also
valid if the Actor decodes it into the correct actions. The main checks are:

- expected reward reaches the ceiling;
- high-reward action probability approaches one;
- selection probability concentrates on `full_state`;
- `full_state` has the highest per-Observer expected reward;
- `full_state` messages are separated across the three target actions.

## 5.3 Result

The experiment uses 5 seeds, 15000 steps, fixed \(\alpha_\phi=0.05\), fixed
\(\alpha_w=0.005\), and a sweep over \(\alpha_\theta\).

| \(\alpha_\theta\) | Final expected | Final sampled | Final action prob. | Final \(P(\text{full})\) | Final message separation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 0.978 | 0.980 | 0.980 | 0.965 | 2.947 |
| 0.05 | 0.984 | 0.985 | 0.985 | 0.971 | 2.494 |
| 0.10 | 0.985 | 0.982 | 0.987 | 0.972 | 2.273 |

All tested Actor learning rates converge across all seeds. The full-state
Observer becomes the highest-reward information source, the Actor's
high-reward action probability rises to the ceiling, and the selection
distribution concentrates on `full_state`.

Generated files:

```text
configs/paper_5_3_joint_imperfect_state.yaml
experiments/run_paper_5_3.py
docs/figures/paper5_3_joint_imperfect_state.pdf
docs/figures/paper5_3_joint_imperfect_state.png
docs/figures/paper5_3_joint_imperfect_state.svg
outputs/paper_5_3/joint_imperfect_state_curves.csv
outputs/paper_5_3/joint_imperfect_state_summary.csv
```
