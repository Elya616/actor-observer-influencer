# Minimal Influencer extension

This extends `CombinedGroupTrainer`, the existing combined message/selection/
reward-feedback loop. Older trainers and experiments remain unchanged.
The example uses three Actors, three Observers, and one shared Influencer.

- Sources `0..N-1` are direct Actors; source `N` is the single Influencer.
- Each Observer selects a source using its existing `SelectionLogits(N + 1)`.
- If it selects the Influencer, the Influencer samples an Actor using its own
  softmax logits and returns that Actor's message unchanged, without detaching.
- Observer source selection and Influencer Actor selection use the Observer's
  reward: loss `-reward * log_probability`. The Influencer selector updates only
  on relay paths. With multiple Observers, their losses are summed as before.
- Actor compression and Observer policy still update only when the existing
  reward report succeeds. No new reward, relay network, or second budget is added.
- The existing Observer following budget now covers `N + 1` sources. The
  Influencer's Actor probabilities sum to one; they are not following rates.
- Evaluation enumerates both paths exactly: effective Actor probability is
  `P(direct Actor j) + P(Influencer) * P(Influencer selects Actor j)`.
- Without an Influencer, keep `SelectionLogits(N)`; the original path is retained.

## Run (from the project root)

```bash
python -m pip install -e '.[dev]'
python experiments/run_influencer.py --steps 500 --debug
python -m pytest tests/test_influencer.py -q
python -m pytest -q
```

## Debug

Pass `debug=True` to the trainer. `trace.debug_rows` records step, state,
Observer, selected source, actual Actor, relay/direct flag, message, action,
reward, and report success. The example prints only the first ten rows.
Checks reject non-finite messages/rewards/gradients, invalid probabilities,
changed relay messages, negative following rates, and exceeded budgets.
Debug does not consume random draws or change the training path.

Tests check identity and gradient preservation, both Actor relay paths,
selected/unselected parameter updates, report gating, direct-path freezing of
the Influencer, learned Influencer selection, exact evaluation, debug-on/off
equivalence, and invalid source dimensions/budgets.

These tests verify implementation invariants, not that adding an Influencer
necessarily improves reward or that this design is the only theoretical model.

## Parameter-sensitivity check

```bash
python experiments/run_influencer.py --steps 2000 --compare-lr --debug
```

`run_experiment()` rebuilds and reseeds the same 3-3-1 setup each time.
`compare_learning_rates()` changes only the Influencer learning rate
(`0.001`, `0.01`, `0.05`), using the same three seeds for each setting.
At each evaluation checkpoint it prints mean expected reward and population
standard deviation across seeds, the mean Influencer probabilities over Actors,
and each Observer's mean probability of choosing the Influencer.
`trace.influencer_actor_probabilities` stores the new probability history.
Use `--influencer-lr 0.05` for an individual run instead.

This is a sensitivity experiment, not a correctness assertion: changing a
learning rate need not improve reward. Matching seeds gives matching initial
conditions; later sampled paths can diverge as learned probabilities change.
