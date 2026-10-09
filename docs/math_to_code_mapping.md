# Math to Code Mapping

| Math symbol or update | Code representation | Notes |
| --- | --- | --- |
| `s_t` | environment state object or tensor | Full state. In early cases this is a dummy vector or one-hot state. |
| `o_t` | actor observation tensor | Equal to `s_t` in Cases 1-2; masked or projected from `s_t` in Case 3. |
| `m_t` | observer message tensor | Produced by `Observer.message(full_state)` in Case 3. Absent in Cases 1-2. |
| `a_t` | sampled action index tensor or int | Drawn from `torch.distributions.Categorical`. |
| `r_t` | scalar reward float or scalar tensor | Looked up from explicit reward table. |
| `pi_theta` | `ActorPolicy`, `SoftmaxPolicy`, or `NeuralPolicy` | PyTorch module producing action logits. |
| `f_phi` | `ObserverMessageModule` | PyTorch module mapping full state to message vector. |
| `theta` | actor policy parameters | Returned by actor/policy `.parameters()`. |
| `phi` | observer message-module parameters | Returned by observer/message-module `.parameters()`. |
| `phi_i` | parameters of observer `i` in Case 4 | Each observer in `MultiObserver` keeps its own `ObserverMessageModule`. |
| `lambda` | following-rate logits in Case 4 | Stored in `FollowingRates.logits`, returned by `following_rates.parameters()`. |
| `rho = softmax(lambda)` | following-rate probabilities in Case 4 | `FollowingRates.forward()` returns differentiable weights; `current_weights()` returns detached weights for logging. |
| `Categorical(rho)` | observer-selection distribution in Case 4 | `FollowingRates.distribution()` returns a `torch.distributions.Categorical` whose `log_prob(k)` keeps the autograd path into `lambda`. |
| `k_t ~ rho` | sampled observer index | `MultiObserver.sample_observer_and_message` returns `(k_t, m_t, log rho_{k_t})`. |
| `log pi_theta(a_t | input)` | `distribution.log_prob(action)` | Use categorical distribution API after logits/probs are built. |
| `theta` update | `loss.backward()` and actor optimizer step | Minimize `loss = -reward * log_prob` to implement reward ascent. |
| `phi` update | backprop through message into observer parameters | In Case 3, actor input must preserve autograd path from `m_t = f_phi(s_t)`. |
| `phi_{k_t}` update | backprop through the selected observer's message | In Case 4, only `observers[k_t].message(s_t)` enters the forward graph, so only `phi_{k_t}` receives a gradient on this step. |
| `lambda` update | REINFORCE on the discrete selection | In Case 4, the loss includes `-reward * log rho_{k_t}`, which is the score-function estimator for `grad_lambda E_{k ~ rho}[r]`. |
| `w` | paper Section 5.1 Observer-selection logits | Stored in `FollowingRates.logits` when `StagedMultiObserverTrainer` is used for fixed-Observer selection. |
| `lambda_i(w)` | paper Section 5.1 Observer-selection probability | Returned by `FollowingRates.current_weights()` for logging and by `FollowingRates.distribution()` for sampling. |
| fixed `f_phi_i` | controlled Observer message table | Implemented by `FixedTableMessageModule`, which maps one-hot state vectors to fixed scalar messages. |
| fixed `pi(. | m)` | controlled Actor decoder | Implemented by `SumDecoderPolicy`, which maps scalar messages near `0, 1, 2` to actions `0, 1, 2`. |
| input-masked `f_phi_i` | learnable Observer message with fixed available features | Implemented by `MaskedObserverMessageModule`; used in paper Section 5.2 to distinguish no-state, partial-state, and full-state Observers. |
| `A_t = r_t - b_t` | Section 5.1 optional advantage multiplier | Enabled with `selection_update_rule="advantage_global"` in `StagedMultiObserverTrainer`. The default remains `raw_reward`. |
| `b_t` | global reward baseline | Stored as `StagedMultiObserverTrainer.selection_baseline` and updated by an exponential moving average after each step. |
| active Actor `j_t` | `GroupImperfectStateTrainer._sample_actor_index()` | Paper Section 7 samples one Actor per step before Observer selection. |
| `p_j` | `actor_selection_probs` | Exogenous probability that Actor `j` is active in Section 7. |
| `w_j` | `following_rates_by_actor[j].logits` | Each Actor has its own Observer-selection logits in Section 7. |
| `lambda_{j,i}` | `following_rates_by_actor[j].current_weights()[i]` | Actor-specific probability of selecting Observer `i`. |
| `k_t ~ lambda_{j_t}` | `GroupImperfectStateTrainer._sample_message(actor_index, full_state)` | Samples an Observer from the active Actor's selection distribution. |
| `theta_j` | parameters of `actors[j]` | In Section 7.3 only the active Actor's policy optimizer steps. |
| `A_t = r_t - b_{j_t}` | Section 7 optional per-Actor advantage multiplier | Enabled with `selection_update_rule="advantage_per_actor"` in `GroupImperfectStateTrainer`. It applies only to the selection-logit term. |
| `b_j` | per-Actor reward baseline | Stored in `GroupImperfectStateTrainer.selection_baselines[j]` and updated only when Actor `j` is active. |
| `H(lambda_j)` | selection entropy | Computed by `GroupImperfectStateTrainer._selection_entropy(j)`. A positive `selection_entropy_beta` discourages early collapse of the active Actor's Observer-selection distribution. |
| `p_j lambda_{j,i}` exposure | empirical Actor/Observer selection exposure | Computed in `experiments/run_paper_7_collapse_analysis.py` from `selected_actor_indices` and `selected_observer_indices`. This measures how often Actor `j` gives Observer `i` a chance to send messages and receive compression updates. |
| compression-gradient attribution | `GroupImperfectStateTrainingTrace.observer_gradient_norms` | Per-step gradient norms for each Observer's message parameters, recorded after `loss.backward()` and before optimizer steps. The collapse-analysis script aggregates these by active Actor and Observer. |

## Case Inputs

Case 1 actor input:

```text
input_t = tensor([1.0])
action_t ~ pi_theta(. | input_t)
reward_t = reward_table[action_t]
```

Case 2 actor input:

```text
state_t ~ D
input_t = one_hot(state_t)
action_t ~ pi_theta(. | input_t)
reward_t = reward_table[state_t, action_t]
```

Case 3 actor input:

```text
state_t ~ D
observation_t = projection_or_mask(state_t)
message_t = f_phi(state_t)
input_t = concat(observation_t, message_t)
action_t ~ pi_theta(. | input_t)
reward_t = reward_table[state_t, action_t]
```

Case 4 actor input:

```text
state_t ~ D
observation_t = projection_or_mask(state_t)
rho_t = softmax(lambda)
k_t ~ Categorical(rho_t)              # discrete observer selection
message_t = f_phi_{k_t}(state_t)       # only the selected observer's message
input_t = concat(observation_t, message_t)
action_t ~ pi_theta(. | input_t)
reward_t = reward_table[state_t, action_t]
```

Paper Section 5.1 actor input:

```text
state_t ~ D
lambda_t = softmax(w)
k_t ~ Categorical(lambda_t)           # observer selection
message_t = fixed_f_phi_{k_t}(state_t)
action_t ~ fixed_pi(. | message_t)
reward_t = reward_table[state_t, action_t]
```

Paper Section 5.2 actor input:

```text
state_features_t = tensor([x_1, x_2])
lambda_t = softmax(w)
k_t ~ Categorical(lambda_t)
message_t = f_phi_{k_t}(mask_{k_t} * state_features_t)
action_t ~ fixed_pi(. | message_t)
reward_t = reward_table[state_t, action_t]
```

Paper Section 5.3 actor input:

```text
state_features_t = tensor([x_1, x_2])
lambda_t = softmax(w)
k_t ~ Categorical(lambda_t)
message_t = f_phi_{k_t}(mask_{k_t} * state_features_t)
action_t ~ pi_theta(. | message_t)
reward_t = reward_table[state_t, action_t]
```

Paper Section 7 actor input:

```text
state_t ~ D
j_t ~ Categorical(p)                         # one active Actor
lambda_{j_t} = softmax(w_{j_t})
k_t ~ Categorical(lambda_{j_t})              # active Actor's Observer choice
message_t = f_phi_{k_t}(state_t)             # shared selected Observer
action_t ~ pi_{theta_{j_t}}(. | message_t)
reward_t = aggregated_reward_table[state_t, action_t]
```

## Losses

Actor-only cases:

```text
loss_actor = -r_t * log pi_theta(a_t | input_t)
```

Partial-observation case:

```text
message_t = f_phi(s_t)
actor_input_t = build_input(o_t, message_t)
loss = -r_t * log pi_theta(a_t | actor_input_t)
```

If actor and observer optimizers both receive `loss.backward()`, PyTorch applies
the chain rule:

```text
grad_phi log pi_theta(a_t | o_t, f_phi(s_t))
  = grad_m log pi_theta(a_t | o_t, m_t) dot grad_phi f_phi(s_t)
```

Multi-observer case (discrete selection):

```text
rho_t = softmax(lambda)
k_t  ~ Categorical(rho_t)
message_t = f_phi_{k_t}(s_t)
actor_input_t = build_input(o_t, message_t)
log_rho_t = log rho_t[k_t]
loss = -r_t * (log pi_theta(a_t | actor_input_t) + log_rho_t)
```

A single `loss.backward()` reaches three parameter sets through three
independent paths:

```text
grad_theta:    via log pi_theta(a_t | o_t, m_t) only
grad_phi_{k_t}: via m_t = f_phi_{k_t}(s_t); other phi_i receive no gradient
grad_lambda:   via log rho_t[k_t] (REINFORCE on the discrete selection)
```

This matches the PDF Case 4 update rule that only updates the parameters of
the sampled observer plus the actor and the following-rate logits.
Evaluation marginalises over both random choices so partially-collapsed
following rates are scored fairly:

```text
E[r | s] = sum_k rho_t[k] * sum_a pi_theta(a | o, f_phi_k(s)) * r(s, a)
```

Paper Section 5.1 fixed-Observer selection:

```text
lambda_t = softmax(w)
k_t ~ Categorical(lambda_t)
message_t = fixed_f_phi_{k_t}(s_t)
action_t ~ fixed_pi(. | message_t)
loss = -r_t * log lambda_t[k_t]
```

Only the selection logits update in this stage. The staged trainer enforces
this by receiving a `following_optimizer` but no `actor_optimizer` and no
`observer_optimizer`.

Paper Section 5.1 optional global-advantage ablation:

```text
lambda_t = softmax(w)
k_t ~ Categorical(lambda_t)
message_t = fixed_f_phi_{k_t}(s_t)
action_t ~ fixed_pi(. | message_t)
A_t = r_t - b_t
loss = -A_t * log lambda_t[k_t]
b_{t+1} = (1 - beta) * b_t + beta * r_t
```

This is not the default paper update. It is used only by
`experiments/run_paper_5_1_advantage_ablation.py` to compare the raw reward
multiplier with a standard global baseline. The baseline is global, not
Observer-specific, so it does not encode the sampled Observer identity.

Paper Section 5.2 selection plus selected-Observer compression:

```text
lambda_t = softmax(w)
k_t ~ Categorical(lambda_t)
message_t = f_phi_{k_t}(mask_{k_t} * s_t)
action_t ~ fixed_pi(. | message_t)
loss = -r_t * (log lambda_t[k_t] + log pi(action_t | message_t))
```

Only `w` and `phi_{k_t}` update. The Actor decoder has no optimizer. The staged
trainer enforces selected-only Observer learning because only the selected
Observer's message is built before `loss.backward()`.

Paper Section 5.3 selection, compression, and Actor policy:

```text
lambda_t = softmax(w)
k_t ~ Categorical(lambda_t)
message_t = f_phi_{k_t}(mask_{k_t} * s_t)
action_t ~ pi_theta(. | message_t)
loss = -r_t * (log lambda_t[k_t] + log pi_theta(action_t | message_t))
```

All three optimizers are present in `StagedMultiObserverTrainer`: the
following-logit optimizer, the Observer optimizer, and the Actor optimizer. The
selected Observer is still the only Observer in the autograd graph.

Paper Section 7.1 group Observer selection:

```text
j_t ~ Categorical(p)
lambda_{j_t} = softmax(w_{j_t})
k_t ~ Categorical(lambda_{j_t})
message_t = fixed_f_phi_{k_t}(s_t)
action_t ~ fixed_pi(. | message_t)
loss = -r_t * log lambda_{j_t}[k_t]
```

Only the active Actor's selection logits update. In code this is represented by
one `FollowingRates` object per Actor. The shared following optimizer contains
all Actor-specific logits, but only `following_rates_by_actor[j_t]` appears in
the sampled log-probability, so inactive Actors receive no gradient.

Paper Section 7.2 group selection plus selected-Observer compression:

```text
j_t ~ Categorical(p)
lambda_{j_t} = softmax(w_{j_t})
k_t ~ Categorical(lambda_{j_t})
message_t = f_phi_{k_t}(s_t)
action_t ~ fixed_pi(. | message_t)
loss = -r_t * (log lambda_{j_t}[k_t] + log pi(action_t | message_t))
```

`GroupImperfectStateTrainer` calls only `observers[k_t].message(full_state)`.
This preserves the selected-only compression update: `phi_{k_t}` receives the
message gradient and all other Observer parameters stay unchanged on that
step.

Paper Section 7.3 group selection, compression, and active-Actor policy:

```text
j_t ~ Categorical(p)
lambda_{j_t} = softmax(w_{j_t})
k_t ~ Categorical(lambda_{j_t})
message_t = f_phi_{k_t}(s_t)
action_t ~ pi_{theta_{j_t}}(. | message_t)
loss = -r_t * (log lambda_{j_t}[k_t] + log pi_{theta_{j_t}}(action_t | message_t))
```

The trainer receives a list of per-Actor optimizers. During `loss.backward()`
the active Actor's policy, the active Actor's selection logits, and the
selected Observer's compression are in the graph. The step routine only calls
`actor_optimizers[j_t]`, so inactive Actor policies do not update.

Paper Section 7 stabilization variants:

```text
j_t ~ Categorical(p)
lambda_{j_t} = softmax(w_{j_t})
k_t ~ Categorical(lambda_{j_t})
message_t = f_phi_{k_t}(s_t)
action_t ~ pi_{theta_{j_t}}(. | message_t)

selection_multiplier_t =
    r_t                         # raw_reward, the paper default
    r_t - b_{j_t}               # advantage_per_actor ablation

loss_selection =
    -selection_multiplier_t * log lambda_{j_t}[k_t]
    - beta * H(lambda_{j_t})

loss_action_and_compression =
    -r_t * log pi_{theta_{j_t}}(action_t | message_t)
```

The advantage and entropy terms are stabilization checks. They do not change
which sampled Actor, Observer, message, action, and reward are used in a step.
They only change how the active Actor's selection logits are updated. The
selected Observer's compression and the active Actor's policy still receive the
raw-reward policy-gradient signal through the action log probability.

Paper Section 7 collapse-analysis diagnostics:

```text
active_count[j] =
    number of steps where Actor j was active

selection_exposure[j, i] =
    number of steps where Actor j selected Observer i
    / active_count[j]

Q[j, i] =
    expected reward if Actor j always used Observer i
    at the current parameters

gradient_share[j, i] =
    cumulative compression-gradient norm from Actor j to Observer i
    / cumulative compression-gradient norm from Actor j to all Observers
```

These diagnostics separate three questions that average reward can mix
together: whether an Actor selects the full Observer, whether the full Observer
is actually better for that Actor, and whether that Actor gave the full
Observer enough compression-learning signal.

## Implementation Guardrails

- Keep reward tensors scalar before multiplying by `log_prob`.
- Do not detach observer messages in Case 3 training or the selected
  observer's message in Case 4 training.
- In Case 4, only call `observers[k_t].message(s)` after the sample, never
  build messages for all observers in the training step. Building all
  messages would put every `phi_i` in the autograd graph and contradict the
  PDF update rule.
- In Section 7, keep the distinction between Actor-specific selection logits
  and shared Observer compression: `w_j` is per Actor, but `phi_i` belongs to
  Observer `i` and is shared across Actors.
- In Section 7.3, do not step inactive Actor optimizers. The update is for
  `theta_{j_t}`, not for every Actor policy.
- Keep `selection_update_rule="raw_reward"` as the default for paper Section
  5.1. The advantage update is an ablation and should be clearly labeled as
  such in plots and tables.
- Keep `selection_update_rule="raw_reward"` as the default for paper Section
  7. The per-Actor advantage and entropy options are stabilization diagnostics,
  not replacements for the paper update.
- Do not use a per-Observer baseline as the main policy-gradient baseline,
  because that would depend on the sampled Observer identity and change the
  estimator.
- Do not use `argmax` during stochastic training unless evaluating a greedy
  policy.
- Track logits and probabilities in tests to detect NaNs early.
- Make the optimal action table explicit so convergence checks are meaningful.
