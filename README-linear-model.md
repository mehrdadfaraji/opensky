# GA Accident Fatality Prediction — Attempt 1 (failed model, documented)

**Goal:** predict whether a general-aviation accident will be fatal, using only the weather
conditions at the time (visibility, cloud ceiling, wind, gusts, temperature, general weather
condition) — sourced from the NTSB's public accident database (CAROL / `avall.mdb`),
30,292 historical accidents.

**Status:** this first model does not work. Documenting why, honestly — a "here's what I
tried, it failed, and here's the actual reason" writeup is more useful (and more honest)
than only ever showing code that happens to work.

## The setup

- 30,292 historical GA accidents, 20.7% marked fatal.
- 7 features: sky ceiling, visibility, wind speed, gust speed, temperature, and a
  one-hot-encoded basic weather condition.
- Model: a single linear combination of these features (no neural network yet) —
  `prediction = coefficients · features`.
- Trained with gradient descent, using the average absolute difference between prediction
  and actual outcome (L1 / MAE loss) as the error signal.

## Why it doesn't work

In plain terms: this setup asks the model to output a single number that's supposed to land
on exactly 0 (survived) or exactly 1 (fatal), but nothing forces that number into a sensible
0–1 range — it can output negative numbers, or numbers that never get close to 1 either way.
On top of that, the error signal used to correct the model (average absolute difference)
pushes just as hard whether the model is close to right or wildly wrong — there's no
"easing off" as it gets closer to the answer, so training bounces around a bad answer
instead of settling into a good one.

Concretely: guessing "no one dies" for every single accident, with zero training, already
beats this trained model on its own error metric (0.2066 vs. this model's 0.24–0.32). That's
the clearest sign something in the setup — not just the amount of training — is broken.

See the `## Postmortem` section at the end of `ntsb-accident-pred.ipynb` for the full
technical breakdown (the gradient math and the learning-rate sweep that confirms this).

## What's next

The fix is architectural, not "more epochs": bound the model's output to a real probability
with a sigmoid function, and switch the error metric to binary cross-entropy, which
naturally slows down near the right answer and pushes hard when the model is confidently
wrong. That version is being built next, as its own commit on top of this one.
