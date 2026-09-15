# Examples

## `end_to_end.py`

Runs the whole platform on a small configuration in about a minute: generate,
build features, train five models, evaluate the promotion gates, stream the test
window through the broker and consumer, and produce a drift report. No network,
no credentials, no external service.

```bash
python examples/end_to_end.py
```

Worth watching for:

- **ROC-AUC barely separates the models; PR-AUC does.** This is the whole reason
  PR-AUC leads everywhere in this repository.
- **The promotion gates reject a 0.002 regression.** The example deliberately
  offers the gates a slightly worse challenger to show the refusal naming the
  gate and the numbers.
- **The consumer is warmed before serving.** A cold store scores measurably
  worse until its windows fill — quantified in `docs/results.md`.
- **Streamed and batch PR-AUC land close together**, which is the training and
  serving paths agreeing.
