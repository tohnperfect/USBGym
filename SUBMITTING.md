# Adding a method to the leaderboard

The [leaderboard](LEADERBOARD.md) lists every published method that was plugged into a slot of USB Gym,
with the per-episode values behind each row, and the methods added since. This page is the request path
for a new method: write a plug-in, pass the admission gate, run the unchanged plug protocol, check the
result file, then open the leaderboard form on GitHub and paste the file. For a slot whose condition
needs an artifact that is not released, paste the gate record instead, and the maintainers run the
plug-in. The method's DOI becomes its reference in the methods list.

## What a row means

Every row follows the plug protocol (`usbg.leaderboard.protocol`): one slot is swapped at a time and the
rest of the loop stays at that slot's condition. A method meets the same 60 held-out episodes as the slot's
reference component (six held-out patients, two targets and five screened windows each, evaluation seed
999, contact on, the windows file shipped in `src/usbg/data/`), at 12 steps. Training uses the 13 training
patients only, every setting is fixed before any held-out run, reinforcement learners get 57,600
environment steps and imitation learners 40,000 samples, learned methods run seeds 0, 1 and 2 (a method
that trains nothing may run seed 0 only and is then paired with seed 0 of the reference), and plugged
networks run batch normalization in evaluation mode.

The row is scored as the published rows are: final translation error (mean over seeds of patient means)
and its difference from the slot's reference component, paired by seed, with the 95% paired t CI, the
patients of 6 and the seeds of 3 sharing its sign, and the post hoc TOST at ±1 mm. A row with three seeds
also gets the patient-by-seed bootstrap interval of Appendix B of the article (patients resampled, then
seeds within each patient, the seed index paired with the reference's; 10,000 resamples, seed 0). A new
method belongs to none of the pre-specified families, so it is marked post hoc: its paired t p is shown
unadjusted, with no Holm adjustment and no pre-specified decision word ("post hoc"; "secondary" for a
privileged method, as in Table 5). A seed-0 row is labelled as seed 0 against the reference's seed 0.
The rows from the article keep their Holm p and decisions.

The leaderboard is not a ranking. Each row was tested against its slot's reference, not against the
other rows, and where the decision is no detectable difference, the methods are not distinguished from
the reference or from each other.

## Slots, plug-ins and who runs them

| Slot | The plug-in's factory returns | Condition (slot header of the leaderboard) | Runs |
| :-- | :-- | :-- | :-- |
| `renderer` | `make(device)`: `render(label_slice, *, device, cfg, seed) -> float32 [0, 1]` | frames from the plugged renderer, no texture stage; CT mask†; the physics-trained policy, residual on; seeds 0 to 2 | you, with the release assets |
| `texture` | `make(device)`: a callable `img01 -> img01` (`wants_label = True` for a label-conditioned stage) | frames from the plugged stage; mask from that stage's own matched U-Net; servo base, residual off; seeds 0 to 2 | maintainers for now (see below) |
| `mask` | `make(device)`: an object with `segment(img01) -> int64 mask` | SonoSPADE frames; mask from the plugged source; servo base fitted under that mask, residual off | maintainers |
| `ctrl` | `make(seed, device)`: `reset(ep, env)`, `act(obs, info)`, optional `train(make_env, budget, seed)` | SonoSPADE frames; CT mask†; the plugged controller; seeds 0 to 2 | maintainers |
| `reward` | `make()`: a `usbg.plug.rewards.RewardPlug` with `potential(cur_img, cur_lab, tgt_img, tgt_lab, **kw)` | SonoSPADE frames; CT mask†; the REINFORCE residual from the imitation start, trained with the plugged potential | maintainers |
| `anatomy` | label volumes of the same 19 CTs | five reference levels of the oracle ladder rerun on the plugged labels | maintainers |

† Privileged: simulator state that no physical rig provides. A controller that is not declared
privileged never receives `info["privileged"]` during the held-out evaluation.

The mask, controller, reward and anatomy conditions need the SonoSPADE stage checkpoint of the reported
runs (and, for the controller and reward slots, the reference policies trained on its frames). Its real
training frames come from a public set whose terms for redistributing derived weights are not
established, so these artifacts are not released (`src/usbg/leaderboard/data/assets.json`, "not_released").

The texture slot trains the stage-matched U-Net of each seed with the reference recipe, which uses the
annotated Vitale et al. frames and the KidneyUS training split (README, Data preparation). KidneyUS
needs registration with its maintainers and is licensed CC BY-NC-SA, so the U-Net you train inherits
those terms: do not redistribute it. The recipe also needs the fixed KidneyUS test assignment of the
reported runs (`segtrain/data/real/kidneyus_split.npz`), which is not yet published; until it is, the
runner stops with that message, and the texture slot goes the maintainer-run way.

For a maintainer-run slot, pass the gate yourself (step 2), then open the form with the request type
"Maintainer run" and paste the gate record. The workflow checks the record (gate version, admitted, the
form's slot, the plug-in entry and its source sha256) and answers "admitted; queued for a maintainer
run". The maintainers run your plug-in under the protocol from its code URL and post the result.

## Steps

Run every command from the root of a clone, after [Install](README.md#install) with an editable install
(`pip install -e .`: the runner uses the clone's `scripts/` and `segtrain/` folders) and
[Data preparation](README.md#data-preparation).

1. **Start from a template.** Each slot has one in `src/usbg/leaderboard/templates/`:

   ```bash
   python -m usbg.leaderboard template --slot renderer --out my_renderer.py
   ```

   Edit it; keep the factory `make`. Its module path is the plug-in entry, `my_renderer:make`.

2. **Pass the admission gate.** It runs on two training patients only, with CUDA watched: a non-privileged
   plug-in must give identical outputs with its privileged inputs removed or poisoned, and, where Apple MPS
   is available, its CPU and MPS outputs must agree within fixed tolerances.

   ```bash
   CUDA_VISIBLE_DEVICES= USBG_SEAT=anatomical USBG_LEARNED_SEG=0 \
       python -m usbg.leaderboard check --slot renderer --plugin my_renderer:make --out gate.json
   ```

   In PowerShell, set the variables first:

   ```powershell
   $env:CUDA_VISIBLE_DEVICES = ""; $env:USBG_SEAT = "anatomical"; $env:USBG_LEARNED_SEG = "0"
   python -m usbg.leaderboard check --slot renderer --plugin my_renderer:make --out gate.json
   ```

   Add `--privileged` if the plug-in reads simulator state; the gate must then catch it doing so.

3. **Fetch the reference artifacts** (renderer slot): the physics-trained reference policies of seeds 0 to
   2, downloaded from the release and checked against their sha256 (the runner checks them again).

   ```bash
   python -m usbg.leaderboard fetch --dest assets
   ```

4. **Run the protocol.** `--dry-run` prints every step and command first. The runner writes
   `submission.json`, with the per-episode values, the summary, the protocol settings and the provenance
   (package versions, device, wall time, the sha256 of the plug-in's source, and the commands with paths
   relative to the repository root, so the file names no local folder).

   ```bash
   python -m usbg.leaderboard run --slot renderer --plugin my_renderer:make \
       --name "My renderer" --doi 10.1234/abcd.5678 --code-url https://github.com/<you>/<repo> \
       --license MIT --description "One sentence on the method." --assets-dir assets --out submission.json
   ```

   The name, license and description are plain text: one line, without `|`, `@`, `[`, `]`, `<`, `>` or
   backticks. `--device cpu` (or `mps`, `cuda`) sets the device of the plug-in and of the evaluation. The
   renderer slot trains nothing and evaluates in minutes on a laptop. `--patients s0358` runs a smoke test
   on one held-out patient's 10 episodes; its file is marked partial and is not a valid result.

   The DOI is the bare DOI of the method's publication, without `https://doi.org/`. A work whose venue
   assigns no DOI (for example PMLR, JMLR or ICLR proceedings) gives the DataCite DOI of its arXiv version,
   `10.48550/arXiv.<id>`.

5. **Check the file.** The validator never imports your code: it checks the type of every section, the
   schema and protocol version, the method's text fields, the DOI pattern, the code URL, the protocol
   settings and the 60 episodes of every seed, refuses absolute local paths in the provenance, and
   recomputes the summary from the per-episode values against the slot's reference values shipped with
   the package.

   ```bash
   python -m usbg.leaderboard validate submission.json
   ```

6. **Open the leaderboard form** (GitHub, New issue, "Leaderboard submission"): method name, slot,
   request type, DOI, code URL and license, then paste the whole `submission.json` (or, for a maintainer
   run, the gate record) and tick the two checks. The pasted file becomes public with the issue. A workflow
   validates it and keeps one comment on the issue up to date with the row it would add, or the errors;
   edit the issue to rerun it.

## What the maintainers check before adding a row

- The comment of the validation workflow shows a valid file (or an admitted gate record), and the form's
  slot and DOI equal the file's.
- The DOI resolves to the method's publication: `add` fetches its record from Crossref (DataCite for
  arXiv DOIs), refuses it unless the registry title equals the title the maintainer gives with
  `--confirm-title`, refuses a DOI the methods list already holds, and builds the method's reference
  from the registry record. The code URL is public under a license that allows the run.
- The gate admits the plug-in at the code URL's version, and its source sha256 equals the one in the file.
- The maintainers rerun the plug-in from the code URL with the same command (`python -m usbg.leaderboard
  run`; the commands are in the file's provenance) and compare the per-episode values with the pasted ones.
  For the maintainer-run slots this rerun is the result.
- Nothing was tuned on held-out patients: every setting is in the plug-in's code or defaults, fixed before
  the run.

Then

```bash
python -m usbg.leaderboard add submission.json --rerun-by-maintainer --confirm-title "<the publication's title>"
```

merges the row into the leaderboard data and the method, with its registry-checked reference and DOI, into
the methods list, and rewrites `LEADERBOARD.md`; the change is committed with the issue number.
`python -m usbg.leaderboard recompute` and the tests recompute every row, submitted rows included, from
its shipped per-episode values.
