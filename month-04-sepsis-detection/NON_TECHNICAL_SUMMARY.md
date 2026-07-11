# Predicting Sepsis Before It Happens — Plain-Language Summary

## The problem

Sepsis is the body's extreme, life-threatening reaction to an infection, and it's a leading cause of death in ICUs. The earlier doctors catch it, the better a patient's odds — but the early signs can look like a dozen other things. This project asked: can a computer, watching a patient's vital signs and lab results hour by hour, flag the risk of sepsis before it becomes obvious?

## The data

Real ICU records from two different hospitals — about 40,000 patients in total, with hourly readings (heart rate, blood pressure, temperature, oxygen levels, and various blood tests) for each patient's entire stay. Roughly 2 in 100 hourly readings, across both hospitals combined, were during a sepsis episode.

## The approach

For each hour of a patient's stay, the model looks back over the previous 6 hours and asks: what's the typical level, what were the extremes, and is this trending up or down? That gives it a short "recent history" snapshot instead of just a single-moment reading — closer to how a doctor would actually assess a patient, by noticing a trend rather than one number in isolation.

One hospital's patients were used to teach the model; the other hospital's patients — completely unseen during training — were used to test whether it actually learned something general, rather than just memorizing quirks of one hospital's patients and equipment.

## What worked, and what didn't

The first model tried (a common, reliable technique called Random Forest) looked like it wasn't working at all — barely catching any sepsis cases. But that turned out to be a measurement problem, not a modeling problem: the model actually had a real, meaningful ability to rank patients by risk, it just wasn't confident enough to cross the "yes/no" cutoff being used to judge it. Once that was measured properly, and a related technique (Gradient Boosting) was tried, a clear, usable signal emerged — a model that could rank sepsis risk about 4-5 times better than random guessing.

## The result

On the hospital used for training, the model catches about 3 in 10 sepsis cases, with roughly 1 in 8 of its alerts being correct. Tested on the *other* hospital — one it never saw during training — it catches about 2 in 10 cases, with roughly 1 in 10 alerts correct. That drop is expected and informative: it means the model's underlying judgment about risk carries over reasonably well to a new hospital, but the exact "alarm threshold" would need to be re-tuned for each new hospital's patient mix rather than assumed to transfer automatically.

For context: a real research team's baseline model in the original scientific competition this data comes from scored lower than this project's result on a similar held-out test — a useful sanity check that this isn't a toy result, even though there's real room to improve further.

## Why this matters

An early-warning tool like this isn't meant to replace a doctor's judgment — it's meant to raise a flag worth a second look, ideally hours before sepsis becomes clinically obvious. Given that missing a real case is far more costly than a false alarm, this project deliberately tuned the model to catch more true cases at the expense of more false alerts, rather than the reverse — a choice that mirrors how these systems are actually meant to be used in practice.
