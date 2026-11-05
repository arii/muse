# Meta Global AI Developer Hackathon — Submission Form Prep

Source: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit
Tab: 733020731 — "Submit Project" modal

Deadline noted on page: Nov 4, 9:00 PM ET

## 0. Project Title
Not a numbered modal field — use wherever the form asks for a title:
`Whole-Arm Grasping, Revisited: 2014 Thesis Meets 2026 Frontier Models`

---

## 1. Team Name*
- Current value: `Onasafari`
- Placeholder: Enter your team name
- Answer:

```
Onasafari
```

## 2. Team Members
- Shown as: Ariel Anders, onasafari
- No input — verify in UI

## 3. Which track are you submitting to?*
Required. Choose one:
- [x] Indie Developer Track
- [ ] Startups Track

Your choice: Indie Developer Track (per SUBMISSION.md + video close card)


## 4. Which problem statement did you build towards?*
Required. Choose one:
- [ ] Build for Your World
- [x] Push the Frontier

Your choice: Push the Frontier (per SUBMISSION.md + video close card)


## 5. Most Viral Track — X post link
Label: "If you're competing for the Most Viral Track, please submit your X post link here to be considered."
Optional.
- Answer URL: (blank — add your X post URL if competing, else leave empty)


## 6. Project Description*
Limit: 0 / 3,000 characters
Placeholder: Describe your project and what problem it solves...
- Draft (paste final below, watch char count):

```
Whole-arm grasping, revisited: my 2014 MIT master's thesis (Learning a Strategy for Whole-Arm Grasping), re-run with 2026 tools. Same spirit — a 3-DOF arm learning enveloping grasps, force closure as the goal — harder task: language-conditioned targets in clutter that I singulate before grasping. Tell the arm what to grasp in words: a VLM grounds the instruction, and SAM 3.1 (Meta Model API) segments the target. An antipodal sampler proposes grasps on the mask, and MuJoCo executes. My thesis ran Box2D with SARSA; here PPO from scratch, BC, and diffusion policies join scripted and VLM planners. I benchmark all seven planner families on identical seeds across five clutter scenes, scored on force closure, singulation, and displacement. Headlines: PPO wins 40/50 by learning restraint; BC ties its teacher 40–40 at 2× data. Docker-reproducible, 115+ commits Oct 27–Nov 4.
```

Char count: 877 / 3000

## 7. Public GitHub Repository*
Placeholder: https://github.com/username/repository
Must be open source per announcements.
- URL: https://github.com/arii/muse (confirmed via git remote; project in `projects/grasping-2026/` — make sure the repo is public before submitting)


## 8. 3-Minute Demo Video*
Placeholder: https://youtube.com/watch?v=...
- URL: https://vimeo.com/1243524488?share=copy&fl=sv&fe=ci (Public, 2:02, verified 200; uploaded Nov 4 by Ariel Anders; share link as pasted in form)


## 9. Describe your use of Meta products to build the submitted project*
Limit: 0 / 3,000 characters
- Draft:

```
SAM 3.1 through the Meta Model API is the perception stage. I POST {base}/v1/responses with model sam-3.1, streaming, one user message carrying a text concept phrase ("red box") plus the camera image. Segmentation arrives as special-token text; I decode it with the official meta_sam_parser package and composite box-relative masks to full-frame. I verified the masks live against simulator ground truth (IoU 0.77; the model saw only the image). I wired the stage into scripts/verify_pipeline.py (segmentation + antipodal grasp synthesis with saved figures) and pinned it with recorded-fixture tests. I built the rest of the system with Muse Code throughout: MuJoCo env, PPO/BC/diffusion training, compare harness, results site, Remotion demo video. Scoping note: the VLM planner baselines run on Nebius (Nemotron); SAM 3.1 perception is the Meta-model component of the system.
```

Char count: 877 / 3000

## 10. Why are Meta's models important to your solution?*
Limit: 0 / 3,000 characters
- Draft:

```
Segmentation is the load-bearing stage: in clutter, boxes touch and overlap, so bounding boxes go ambiguous. Pixel masks keep a single boundary, which is what the grasp sampler reasons over. Text prompting suffices: the concept phrase alone verifies live at IoU 0.77. Muse Code mattered as the build tool: it built the environment, reward studies, training pipelines, eval harness, results site, and video in-session (115+ commits Oct 27–Nov 4). I verified perception live as a stage; closed-loop integration is next.
```

Char count: 517 / 3000

## 11. Provide feedback from your experience using partner technologies.
Limit: 0 / 150 characters
Placeholder: Fireworks, Modal, Baseten
- Draft:

```
No partner tech used — Meta Model API (SAM 3.1) + Muse Code only.
```

Char count: 65 / 150

## 12. How likely are you to build with Meta Model API?*
Required. Choose one:
- [ ] Very likely
- [ ] Somewhat likely
- [ ] Neutral
- [ ] Somewhat unlikely
- [ ] Very unlikely

Your choice: Very likely
Suggested: Very likely (confirm or change)

## 13. How would you describe your overall opinion of Meta Model API?*
Required. Choose one:
- [ ] Very favorable
- [ ] Somewhat favorable
- [ ] Neutral
- [ ] Somewhat unfavorable
- [ ] Very unfavorable

Your choice: Very favorable
Suggested: Very favorable (confirm or change)

## 14. Permission grant checkbox
Text: "By accepting, the submitter(s) grants Meta and Cerebral Valley permission to use, publish, share, and promote the submitted materials across their websites, social media, and other marketing channels."
- [ ] Accepted (you must tick this in the UI yourself)

## 15. How likely are you to recommend the Meta Model API to another developer or engineering team building with LLMs?
Choose one:
- [ ] 0 (not likely)
- [ ] 1
- [ ] 2
- [ ] 3
- [ ] 4
- [ ] 5 (very likely)

Your choice: 5 (very likely)
Suggested: 5 (confirm or change)

## 16. How likely are you to recommend Muse Code to another developer or engineering team?
Choose one:
- [ ] 0 (not likely)
- [ ] 1
- [ ] 2
- [ ] 3
- [ ] 4
- [ ] 5 (very likely)

Your choice: 5 (very likely)
Suggested: 5 — repo developed with it throughout, 115+ commits (confirm or change)

## 17. What was the biggest obstacle your team hit while building this project?
Choose one:
- [ ] Getting access or setting up our account
- [ ] Finding clear documentation or code examples
- [ ] Connecting to our framework or other tools
- [ ] Code or answers from the model that were wrong or not useful
- [ ] The model not following our instructions
- [ ] A model capability we needed was missing
- [ ] Slow responses
- [ ] Errors / outages or unreliable service
- [ ] Rate limits or usage limits
- [ ] Muse Code losing track or stalling on longer tasks
- [ ] Muse Code changing things we didn't ask it to
- [ ] A Muse Code capability we needed was missing (please specify)
- [ ] Something else (please specify)
- [ ] We didn't hit any obstacles

Your choice + specify text if needed: Connecting to our framework or other tools
Suggested: Connecting to our framework or other tools (the Responses SSE stream needed a lane adapter before the official parser accepted it; confirm or change)


## 18. Compared with other AI models your team has used for the same kind of work, how did Meta's models perform on this project?
Choose one:
- [ ] Much better
- [ ] Somewhat better
- [ ] About the same
- [ ] Somewhat worse
- [ ] Much worse
- [ ] We've used other models but couldn't make a clear comparison
- [ ] We haven't used other models for this kind of work

Your choice: We've used other models but couldn't make a clear comparison
Suggested: We've used other models but couldn't make a clear comparison (local SAM is box-prompt, API is text-prompt — different modalities; confirm or change)

## 19. When your team chooses an AI model or coding tool, what matters most?
Choose one:
- [ ] Quality of the answers and code
- [ ] Speed
- [ ] Price
- [ ] Handling long context such as big documents or codebases
- [ ] Agent abilities such as tool calling and multi-step tasks
- [ ] Reliability and uptime
- [ ] Easy integration such as SDKs or an OpenAI-compatible API
- [ ] Fits our editor and workflow
- [ ] Data privacy and terms of use
- [ ] Open weights or self-hosting
- [ ] High enough rate limits
- [ ] Something else (please specify)

Your choice + specify text if needed: Quality of the answers and code
Suggested: Quality of the answers and code (confirm or change)


## 20. What makes it hardest for your team to switch to a different AI model or coding tool?
Choose one:
- [ ] Our product is tuned to our current model's behavior and prompts
- [ ] Testing a new model takes too long
- [ ] Integration work such as new APIs SDKs or formats
- [ ] Our team already knows our current tool
- [ ] Contracts, credits or commitments with our current provider
- [ ] Getting a new vendor approved, such as security, legal or procurement
- [ ] Our data or fine-tuned models are tied to our current provider
- [ ] Features we rely on that other providers don't offer
- [ ] Something else (please specify)
- [ ] Nothing, switching is easy for us

Your choice + specify text if needed: Integration work such as new APIs SDKs or formats
Suggested: Integration work such as new APIs SDKs or formats (custom SSE parsing per provider; confirm or change)


## 21. If your team took this project to production, what would most likely stop you from using Meta's models?
Choose one:
- [ ] Model quality isn't good enough yet for our use case
- [ ] Price or unpredictable costs
- [ ] Rate limits or capacity too low for real traffic
- [ ] Reliability or uptime
- [ ] How our data and our users' data would be stored or used
- [ ] Security or compliance requirements, such as SOC 2
- [ ] Not available in our country or region
- [ ] Not available through our cloud provider, such as AWS, Azure or Google Cloud
- [ ] Missing production features, such as batch processing or spending limits
- [ ] Not sure Meta will keep supporting the API long term
- [ ] Something else (please specify)
- [ ] Nothing would stop us
- [ ] We don't plan to take this project to production

Your choice + specify text if needed: We don't plan to take this project to production
Suggested: We don't plan to take this project to production (research sim benchmark; confirm or change)


---

## Submit
Button: Submit
