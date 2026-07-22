# A personal, embedding-driven 4-class MI scheme (single subject, imperfect cap)

Goal: instead of inheriting the public-dataset classes (hands/feet/tongue), **co-explore
with the model** which imagined tasks are maximally separable *in embedding space* for
**this person on this cap**, and lock in a personal 4-class set.

## 1. Reframe the constraints — most of them help here

| Constraint | Real impact on this goal | Mitigation / why it's OK |
|---|---|---|
| No impedance hardware | Low | Software quality proxy (the viewer panel: per-ch RMS, 50 Hz ratio, railed). Track a per-session channel-quality map. |
| Posterior electrodes don't seat | **Low for MI** | MI lives over **central** cortex (C3/Cz/C4/FC/CP) — those seat fine. Occipital is for vision/alpha, irrelevant to MI. **Design consequence:** pick tasks whose neural generators sit under well-contacted electrodes (central/frontal), and *avoid* tasks that rely on parietal/occipital (e.g. pure visual imagery). |
| Can only record **yourself** | **Actually easier** | Within-subject MI is far easier than cross-subject (0.75 cross → 0.85–0.95 within, with calibration). We sidestep "no other people's data" by using **foundation models** (pretrained on thousands of hours of others) as *frozen, subject-agnostic feature extractors*, and fit only a light within-subject head. |
| Want a personal class set | This is the opportunity | Single subject = we can tailor classes to your brain instead of a population average. |

**Key referencing choice:** use a **small Laplacian** over C3/Cz/C4 (and the FC/CP ring),
not global CAR. A Laplacian is *local* → immune to the bad posterior channels (a bad
electrode only corrupts its own neighborhood, not the central estimate), spatially sharp,
and needs no M1/M2. Keep CAR only for the foundation-model path (which expects it), computed
on the good-channel subset.

## 2. The co-exploration loop ("我和模型一起探索")

```
 screen many tasks → embed every trial → pairwise-separability matrix →
   pick the best mutually-separable 4 → verify across a *different day* →
     neurofeedback co-adaptation → lock the personal 4-class decoder
```

1. **Screen** a candidate repertoire of ~8–12 mental tasks (below), with cued, triggered,
   randomly-interleaved trials. Rest/idle is always one candidate.
2. **Embed** each trial. Two embedding spaces, compared:
   - **Riemannian tangent-space** of the covariance (robust, no training) — primary.
   - **Frozen FM embedding** (CBraMod/LaBraM) — secondary; may capture *non-motor* tasks
     (math, language) that covariance-based features miss.
3. **Separability matrix**: N×N **cross-session** balanced accuracy for every task pair
   (train on day-A trials, test on day-B — never within-block; see §4). This is the
   "difference in embedding space," measured honestly.
4. **Select the best 4** by a **max-min clique**: over all C(N,4) subsets, maximize the
   *minimum* pairwise separability, so all four are mutually distinguishable (not just some).
5. **Stability gate**: re-screen the chosen 4 on a *new day* (fresh cap placement). If
   separability survives a new montage/impedance state, it's neural, not a setup artifact.
6. **Co-adapt**: neurofeedback — show live classifier output; you learn to make the classes
   more separable, the decoder periodically retrains on your newest data. This is the
   "human + model explore together" step (co-adaptive BCI, Millán/Vidaurre style).

## 3. Candidate task repertoire (mix motor + non-motor)

Chosen so generators sit under **well-contacted central/frontal** electrodes:

- **Sensorimotor MI** (mu/beta ERD, central): left-hand, right-hand, both-feet, tongue;
  finer variants if needed — hand *grasp* vs *finger tap*, wrist rotation.
- **Non-motor mental tasks** (often *more* separable, less fatiguing, different networks):
  - Mental arithmetic (serial-7 subtraction) — frontoparietal.
  - Silent verbal fluency (generate words from a letter) — left frontotemporal.
  - Auditory imagery (replay a familiar song) — temporal.
  - Motor-*sequence* imagery (imagine typing / playing piano) — central + SMA.
  - Rest / eyes-open fixation — baseline.

A strong, commonly-separable starting quad is often **{left-hand MI, right-hand MI, mental
arithmetic, rest}** — lateralization gives two classes, math a distinct frontal signature,
rest a baseline — but **let the separability matrix decide**, don't assume.

## 4. Anti-self-deception (the make-or-break for single-subject exploratory)

Flexible class selection on one subject can "discover" separability that is really artifact.
Guard hard:

- **Randomized interleaving** of classes within a session → slow drift can't align with
  labels (kills the "separable because of time/impedance drift" confound).
- **Cross-session, not within-session, evaluation** → the acid test. If a day-A decoder
  fails on day-B, the "separability" was cap-placement, not brain. Report cross-session.
- **Permutation test**: shuffle labels, recompute the separability metric → null
  distribution; the real value must clear it.
- **EMG/EOG audit**: inspect *where/what* the discriminative signal is. If it's >35 Hz or
  frontotemporal, it's muscle/eye, not MI. For a *pure-MI* claim, restrict to sensorimotor
  channels + 8–30 Hz. For a broader *mental-task BCI*, other signatures are fine — but label
  them honestly (a math-vs-rest decoder on frontal theta is a legit mental-task BCI, just not
  "motor imagery"). Tongue/jaw especially → watch temporal EMG.
- **Stillness**: kinesthetic MI (feel the movement, don't do it), eyes fixed.

## 5. Concrete protocol

- **Montage/reference**: all 32 ch acquired; decode from central set
  {FC5,FC1,FC2,FC6,C3,Cz,C4,CP5,CP1,CP2,CP6} with a small Laplacian; per-session drop
  channels the quality panel flags.
- **Trial**: 2 s fixation → cue → **4 s task** → 2 s rest. ~40–60 trials/class/session.
- **Sessions**: ≥2 different days per screening round (for cross-session eval); re-mount the
  cap each day.
- **Preprocessing**: two pipelines kept separate — classic MI (8–30 Hz + Laplacian) and FM
  (0.1–75 Hz + notch + CAR + µV/100, per `FM_input_requirements.md`).
- **Decoder**: within-subject Riemannian TS + logistic-reg / MDM; add per-session
  Euclidean/Riemannian alignment to absorb day-to-day drift.

## 6. Maps onto tools we already have (built + verified)
- **Screening cues + markers** → `src/experiment/screening_paradigm.py` (PsychoPy 10-task
  cue + LSL 'MI_Cues' markers; `--simulate`/`--dry-run` need no PsychoPy). Record with
  LabRecorder alongside the cap's 'Cap32' stream → XDF.
- Quality/channel selection → the viewer's signal-quality panel (`src/acquisition/viewer.py`).
- Embeddings + separability + selection + **permutation test** →
  `src/exploration/task_separability.py --embedding {tangent,cbramod,labram} --permutations N`.
  Both a Riemannian and a frozen-FM embedding space, so you can see which one best separates
  *your* repertoire (FM may win for non-motor tasks that covariance misses).
- Session alignment → `src/baselines/riemann_alignment.py` (EA), applied cross-session.

Validated on 2a subj-1 (cross-session, 4 motor classes): tangent mean-sep **0.92** (p=0.005),
CBraMod-embedding **0.75** (p=0.010) — both beat shuffled-label null (~0.50). Riemannian wins
on pure motor MI; the two spaces also disagree on *which* pair is hardest (tangent: feet↔tongue;
CBraMod: left↔right), so agreement across both is a robustness signal.

## 7. Open decisions (yours)
- Pure MI only, or MI + non-motor mental tasks? (I recommend **mixed** — more separable.)
- How many days/sessions can you realistically record? (Sets statistical power.)
- Offline-first (screen → select → train) or straight to co-adaptive neurofeedback?
  (Recommend offline-first to find the classes, then neurofeedback to sharpen them.)
