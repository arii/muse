---
id: feels
title: Feels — Teaching a Robot to Feel
---

# Feels — Teaching a Robot to Feel

## What is in the cup?

<!-- visual: title -->

What is in the cup? This robot cannot see inside it — and it does not need to. Frontier SF — made by Ariel Anders using obvious.ai.

## The swirl is the sensor

<!-- visual: split -->

Feels teaches a robot to feel. The arm picks up a sealed, lidded cup, swirls it, and reads the joint torques.

- Pick, swirl, place — one macro
- Torques become a 10-dimensional physical signature
- Vector search retrieves the nearest taught example
- Teach a new item: one label, one swirl, no retraining
- No force-torque sensor, no camera on the contents

## A mystery cup, live

<!-- visual: terminal -->

Here is the real CLI on a recorded mystery cup — contents unknown to the model.

```bash
$ python src/predict_content.py recordings/recording_20261029_200617_mystery.csv
file: recording_20261029_200617_mystery.csv
true: mystery -> mystery   (swirl in recording: True)

[stage 1] loaded   (loaded=0.77, empty=0.23)
[stage 2] liquid   (liquid=0.98, solid=0.02)

PREDICTION: LIQUID  (swirl-model agrees: liquid)
```

Stage one: the cup is loaded. Stage two: liquid, with the swirl model in agreement. Every number comes from torques alone.

## Reading the spectrum

<!-- visual: terminal -->

The same signature drives a state spectrum, with the nearest neighbors from the Atlas vector library. Loadedness and rigidity, read at a glance.

```bash
$ python src/state_spectrum.py recordings/recording_20261029_200617_mystery.csv --neighbors

  STATE SPECTRUM
   loadedness     empty [███████████████████·····] full        77%
   rigidity      liquid [························] solid        2%

  reads as: LIQUID  (loaded 77%, solid 2%)

    0.949  water_075pct
    0.904  water_050pct
```

The neighbors are all water — the mystery cup reads as water, at point nine cosine similarity.

## The numbers

<!-- visual: bullets -->

- 115 labeled telemetry recordings, 5 content classes
- 10-D signatures embedded on the unit hypersphere
- Leak-free time-split evaluation: earliest 70% train, latest 30% test
- Static-hold baseline: only 3 of 7 correct

And that last number is the honest one: static hold torques are identical across contents. The swirl dynamics are the innovation.

## Tactile common sense

<!-- visual: title -->

Tactile common sense, on hardware you already own. Feels — built by Sree Dhyuti Nimmagadda and Ariel Anders at the Mongo Learn robot hackathon.
