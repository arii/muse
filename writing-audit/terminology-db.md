# Terminology database for writing audit

> `terminology-db.json` is the canonical source scripts read. Each rule has an
> id, category, regex flag, case sensitivity, severity (`error` = must-fix:
> banned terms, prompt leakage, banned subtitles; `warning` = review), a
> suggestion, and a message. This file is the human-readable companion —
> keep them in sync.

Each entry flags a sentence for review. Flag means review, not auto delete. Ari decides.

## Corporate and AI transition filler
- In today's fast-paced digital landscape
- In today's rapidly evolving world
- It's important to note that
- It is worth noting that
- In conclusion (when restating)
- At the end of the day

## Redundant explanatory restatements
- This allows you to
- This enables you to
- In order to (use "to")
- Due to the fact that (use "because")
- For the purpose of
- In the process of

## Vague marketing modifiers and hyperbole
- deep dive
- unlock
- empower
- supercharge
- revolutionize
- game-changer
- cutting-edge
- world-class
- robust (as vague praise)
- seamless
- leverage (as verb for "use")

## Punchy AI awkwardness
- three-beat punchy lines (e.g. "Fast. Simple. Powerful.")
- cute phrasing or slogans
- startup landing page copy
- dramatic restatements

## Conversational AI metacognition
- Let's dive into
- Let's explore
- As we can see
- As mentioned previously
- It goes without saying

## Overused AI buzzwords
- delve
- tapestry
- testament
- navigate (as metaphor)
- demystify
- multifaceted
- nuanced
- underscore (as "emphasize")
- intricate
- vibrant

## Negative framing markers
Flag and rewrite to state what is true.
- don't, doesn't, didn't
- cannot, can't
- never
- no, not, without
- unlike
- we only use X (use "we use X for")
- no deep learning, no X, non-X

## Hedging and low density
- very, really, quite, rather
- arguably, essentially, basically
- it seems, it appears
- a wide range of, a variety of
- in terms of

## Attention seeking questions
- leading question as paragraph opener (e.g. "Want better results?")
- rhetorical "Did you know?" "What if?"
- question mark headlines that tease instead of state

## Prompt leakage artifacts
- as an AI
- as a language model
- as an AI language model
- my training data
- my knowledge cutoff
- I don't have real-time

## First person in technical docs
Docs use third person system voice ("the system"). Blog posts stay first person.
- we (as authorial voice in docs)
- our team
- us (as authorial voice in docs)

## Banned subtitles
- Why it matters (never needed, delete the section)
- Why this matters

## Sentimental no value sentences
- built together, side by side at the venue (states the obvious, adds no information)
- side by side (as sentimental filler)
- the team came together

## Self referential meta
- the docs you are reading
- this page you are reading
- as you can see here

## Vague totalizers
- Everything here
- All of this
- Everything below

## Vague time markers
- By evening
- Later that day
- At the end of the day (as time marker)

## Trivia identifiers
- unit sfhack12 pattern: internal unit, serial, or asset IDs the reader does not need
- serial number in prose

## Startup jargon
- pivoted (as jargon for changed direction)

## Banned terms (project specific, extend per repo)
- tare -> use baseline
- confound set -> use plain description
- unmeasured weight claims (labels are names, not measurements)
