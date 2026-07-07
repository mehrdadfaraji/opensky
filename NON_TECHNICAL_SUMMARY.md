# Can you predict which plane crashes will be fatal? (plain-language summary)

## The question

The NTSB (the U.S. agency that investigates plane crashes) has kept detailed records on every
general-aviation accident since 1982 — over 30,000 of them. About 1 in 5 of those accidents were
fatal. Given everything known about an accident *before* the outcome — the weather, whether it
happened in daylight or dark, how experienced the pilot was, what kind of plane it was — can a
computer learn to guess in advance which accidents are more likely to be deadly?

This isn't a "build a warning system" project. It's an exercise in a core skill of the field: take
a real, messy dataset, ask a real question of it, and find out honestly whether there's a
learnable pattern in there or not — including reporting it plainly when the first two attempts
mostly didn't work.

## Attempt 1: just the weather — and it failed

The first try used only weather at the time of the accident: cloud height, visibility, wind,
temperature. The technical setup had a subtle flaw (explained in the technical README) that made
the model perform *worse* than simply guessing "nobody died" for every single accident. Rather
than hide that, it's documented as a lesson: sometimes the way you set up the scoring for a model
matters more than how long you train it.

## Attempt 2: fixing the setup, then asking harder questions

Fixing that flaw got a small real result: using weather alone, the model could correctly flag
roughly 1 in 8 fatal accidents in advance. Not impressive, but genuinely better than guessing.

Before trusting that number, it got double-checked by retraining the same model 25 times with
tiny random differences. About 1 in 5 of those training runs turned out to be secretly broken —
they looked fine (correct about 80% of the time overall) but had actually learned to never flag
anything as fatal. That's an important lesson on its own: a model that looks good on the surface
can be quietly useless, and the only way to know is to check more than once.

Then came the more interesting question: is weather really the best clue available, or just the
easiest one to reach for? It turned out not to be. Two things buried in the same free dataset
mattered far more than any weather reading:

- Whether the accident happened **in the dark** — accidents at night in unlit areas were about
  **3 times more likely** to be fatal than ones in daylight.
- **How experienced the pilot was** — once pilot flight-hour history was pulled in from a second
  part of the same database, it turned out to be the single strongest clue of all, ahead of every
  weather variable.

Using this richer information roughly **doubled** how many real fatal accidents the model could
correctly flag in advance — without needing any new data, just paying attention to information
that was sitting right there the whole time.

## Attempt 3: a completely different method — decision trees

The last stage tried a different style of model entirely: instead of a neural network, a
"decision tree" — essentially a flowchart of yes/no questions ("did this happen on airport
property?", "was it dark?") that sorts accidents into groups and checks how often each group
turned out fatal. A "random forest" builds a hundred slightly different versions of that flowchart
and averages their answers, which smooths out the mistakes any single flowchart would make.

This method needed far less fiddling than the neural network to get working, and turned up a
surprise: the single biggest clue wasn't weather or pilot experience at all — it was simply
**whether the accident happened on airport property**. Off-airport accidents were roughly 4 times
more likely to be fatal, which matches real-world intuition (an off-airport accident is more often
a genuine in-flight emergency, while on-airport mishaps tend to be lower-severity).

Depending on how the random forest is tuned, it can be made to catch anywhere from about 1 in 4 to
about 4 in 5 real fatal accidents in advance — but the more it catches, the more false alarms it
also raises. There's no setting that gets both for free; it's a genuine trade-off, not a bug to
fix.

## The honest bottom line

None of these models are good enough to be an actual safety tool — flagging some fraction of
fatal accidents in advance, using only information available before the fact, is a research
finding, not a product. What this project actually demonstrates: a real dataset was cleaned and
questioned rigorously (catching sentinel/error values, catching a real bug that would have
silently hidden an improvement), three different modeling approaches were compared honestly
including a documented failure, and a repeated result was checked more than once before being
trusted. That process — not any single accuracy number — is the actual deliverable.
