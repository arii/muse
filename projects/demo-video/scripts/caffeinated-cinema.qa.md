# QA: Caffeinated Cinema highlight cut

## Provenance

- Script: `scripts/caffeinated-cinema.md`; plan: `src/generated/caffeinated-cinema.json`.
- Caffeinated origin/main: `6488d413a67551d07c73abd41b26535850629295`, archived/built
  outside its unchanged checkout at `/home/ubuntu/caffeinated-cinema-main`.
- Recording: `f8fa53cc-6afd-5e98-9596-0c7b8711e718`, 297 ticks; no mock/invented data.
- Capture URL flags: `?cinema=1&capture=1&recording=f8fa53cc-6afd-5e98-9596-0c7b8711e718`.
- Reused master: `/home/ubuntu/cinema-qa/cinema-source.mp4`, 64 s, 1920×1080, 30 fps.
- Committed downscale: `public/replay/caffeinated-cinema/director.mp4`, 2,446,869 bytes;
  64 s, 1280×720, 30 fps, silent. SHA256:
  `2ce82b3656e2d7df49e34b21f35b92c835084ffb5bb696760e6780c2fafa6992`.
- Proof card: 0.73 boredom at snap, 2 labs founded, 3 post-snap replications,
  1 candidate. Separate pilot table retains “no significant boredom benefit yet”.

## Final ffprobe facts (re-rendered after audit)

File: `/home/ubuntu/repos/muse/projects/demo-video/out/caffeinated-cinema.mp4`.

| Field | Value |
|---|---|
| Format duration / size | 28.715000 s / 8,318,270 bytes |
| Video | H.264, 1280×720, 30/1 fps, 861 frames (28.700 s) |
| Audio | AAC, 48,000 Hz, stereo; AAC padding explains the duration difference |

## Caption sync

Offline pinned Kokoro `af_heart`; sentence cues concatenate exactly to narration.
Measured WAVs round to 132/106/215/198/210 frames. Rechecked audio-fit timing:
`PASS 861 frames, 28.700s; audio-fitted chapter timings and cue text match`.

| Sample | Active cue interval | Visible cue |
|---|---|---|
| 2 s | 0–4.400 | Caffeinated follows a real swarm of artificial scientists. |
| 5.5 s | 4.400–7.933 | Local tweaks give way to rising measured boredom. |
| 9.5 s | 7.933–11.755 | At the recorded snap, new labs begin scouting. |
| 12.5 s | 11.755–15.100 | Quiet stretches move faster, without inventing events. |
| 16 s | 15.100–17.988 | Replicated discoveries lead to the recorded verdict. |
| 19 s | 17.988–21.700 | This is a highlight cut of the same run. |
| 23 s | 21.700–24.763 | The proof card keeps the pilot honest. |
| 26 s | 24.763–28.700 | These small comparisons show no significant boredom benefit yet. |

## Screenshots (outside git)

- `/home/ubuntu/cinema-qa/audit-final-2s.png`
- `/home/ubuntu/cinema-qa/audit-final-5.5s.png`
- `/home/ubuntu/cinema-qa/audit-final-9.5s.png`
- `/home/ubuntu/cinema-qa/audit-final-12.5s.png`
- `/home/ubuntu/cinema-qa/audit-final-16s.png`
- `/home/ubuntu/cinema-qa/audit-final-19s.png`
- `/home/ubuntu/cinema-qa/audit-final-23s.png`
- `/home/ubuntu/cinema-qa/audit-final-26s.png`

All eight re-rendered PNGs hash-identical to the visually inspected pre-refactor
frames: whole source visible, captions separated, real proof card retained.

## Automated audio checks

On the final MP4, `ffmpeg -i out/caffeinated-cinema.mp4 -af volumedetect -vn -sn
-f null -` measured mean **−26.3 dBFS**, max **−7.4 dBFS**. No sample clipping:
max < −0.1 dBFS. Logs: `/home/ubuntu/cinema-volume.txt`.

`ffmpeg -i out/caffeinated-cinema.mp4 -af silencedetect=noise=-35dB:d=0.4 -vn
-sn -f null -` gives these non-silent speech-activity spans (seconds):
**0.000–3.953; 4.736–7.424; 8.242–10.990; 11.621–14.637;
15.426–18.307; 18.850–21.210; 22.000–24.039; 24.576–28.146**.
These are threshold-based activity spans, not recognized-word boundaries.

Against chapter-offset cue windows in the generated plan, speech overlap per
cue is **3.953 / 2.688 / 2.882 / 2.881 / 2.562 / 2.679 / 2.226 / 3.383 s**.
**8/8 cues overlap speech; longest uncued speech = 0.000 s** (<1.5 s).
Logs/check: `/home/ubuntu/cinema-silence.txt`, `/home/ubuntu/cinema-audio-check.py`,
`/home/ubuntu/cinema-audio-check.txt`. Sentence timing remains word-weighted,
not phoneme alignment; contiguous cue coverage does not prove exact word sync.

Local faster-whisper **1.2.1**, unprompted **base.en**, CPU int8, transcribed the
final MP4: **WER 0/63 = 0.00%** against the concatenated narration. WER uses
lowercase alphanumeric tokens and Levenshtein distance, ignoring punctuation
(including extra trailing periods emitted by ASR). Transcript and reference
are in `/home/ubuntu/cinema-audio-check.txt`. No subjective naturalness claim.
