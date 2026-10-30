# Transolver learned slice decomposition

`slice_visualisation.py` reads the slice memberships out of a trained Transolver
checkpoint and draws them, so the decomposition the model learned can be
inspected directly. One file, four subcommands.

## Background

Transolver replaces dense attention over mesh nodes with attention over a fixed
number of learned groups, called slices. For each node, every attention head
produces a soft membership over the slices:

```
w = softmax( W_s x / tau )        shape (batch, heads, nodes, slices)
```

`W_s` is a learned projection onto the slices and `tau` is a learned temperature
that controls how decisive the assignment is. A node's memberships sum to one, so
the assignment is a soft partition of the domain: the node is summarised into the
slices in those proportions, attention is computed between slices rather than
between nodes, and the result is returned to the node as a blend weighted by the
same numbers. Grouping is what makes the cost quadratic in the number of slices
instead of the number of nodes.

Nothing in the training loss refers to these groups. The model is only asked to
predict the deformation, so any correspondence between a slice and a physically
meaningful region of the part is something it arrived at on its own, and has to
be tested rather than assumed.

## Two points that determine whether a figure is meaningful

**Never average over heads.** The projection and the temperature belong to a
single head, so a block with `H` heads computes `H` independent decompositions of
the same mesh. The head average is not a quantity the model computes anywhere,
and averaging several sharp partitions produces something diffuse that resembles
none of them. This script exports and draws a single head throughout.

A useful demonstration: `compare` draws the same slice index on the same frame
for several heads of one block. The geometry, the node set and the slice index are
identical in every panel, so any difference is the learned assignment. On the
press stage at t = 3.0 s, slice 3 of block 7 holds the stationary material under
head 2 and the deforming material under head 3, close to the complement. A slice
index carries no meaning of its own; each head decides independently what to put
in it.

**Sharpness is the wrong criterion for choosing which head to display.** A head
can reach a very high mean leading membership by assigning nearly the whole part
to one slice. A criterion that instead rewards many equally sized groups will
favour a head that cuts the domain into positional bands, which looks structured
but groups by location rather than by behaviour. Neither is what the figure is
meant to show, so the criterion below tests the property directly.

## The selection criterion

**Requirement, applied first.** The blank has two flat end regions, one at each
end, which stay essentially stationary through the stroke. They are physically
identical to one another and lie far apart, so a partition based on physical
state must place both in the same slice, and one based on position cannot. A head
that separates them is rejected regardless of how sharp it is. This is
discriminating in practice: it eliminates a large fraction of the candidates.

**Score among the heads that pass**, averaged over the frames of the stage:

```
flat coherence  x  eta squared
```

- **flat coherence**: the share of the stationary material, taken as vertical
  travel below `FLAT_MM`, held by its single largest slice. A value of 1.0 means
  the whole undeformed blank lands in one slice.
- **eta squared**: the fraction of the variance of vertical travel explained
  between slices rather than within them. A value of 1.0 means every slice is
  internally uniform in physical state. It needs no threshold, which is why it is
  reported next to the threshold-dependent quantity above.

Frames in which the punch has not engaged contain no variance in travel to
explain and are excluded from scoring. That leaves at least one frame available
afterwards as a held-out check: if the decomposition follows the deformation, the
model should use a single slice for the whole blank before contact.

`select` prints every candidate with all of these numbers, so the choice is on
the record, and `--block` and `--head` override it.

## Reading the figures

One panel per slice. The colour at a node is that node's membership in that
slice, on an absolute scale from 0 to 1 shared by every panel and every figure.
Nothing is rescaled per panel and no colour transform is applied, so brightness is
comparable between panels, timesteps and stages.

Because a node's memberships sum to one, a node that is bright in one panel is
dark in all the others. A slice the model uses therefore appears as a bright patch
on a dark part, and a slice it does not use is dark everywhere. **Most panels
being dark is the expected outcome, not a fault of the plotting.**

Two counts are reported and they differ. The number of slices that win at least
one node is larger than the number that are visible, because a slice can win a
node merely by holding the least small value there. The visible count, slices
whose membership exceeds `VISIBLE` somewhere, is the one that describes what a
figure actually shows.

## Usage

Set the paths at the top of the file, or pass them on the command line. `export`
needs the model and the checkpoints; everything else works from the `.npz` files
alone and runs anywhere.

```bash
# 1. capture the memberships. Prints the plan first with --dry-run.
python slice_visualisation.py export --dry-run
python slice_visualisation.py export --td <training_dir> --repo <repo> --out slices

# 2. score every block and head, and see which one the criterion picks
python slice_visualisation.py select --src slices

# 3. draw the grids, one PNG per timestep
python slice_visualisation.py render --src slices --out figures
python slice_visualisation.py render --src slices --stages press --at 3.0
python slice_visualisation.py render --src slices --block 7 --head 2

# 4. the same slice index across several heads, as a check
python slice_visualisation.py compare --src slices --slice 3 --heads 2 3 1 0
```

## What `export` writes

One `.npz` per stage and timestep, plus `index.json`.

| key | shape | meaning |
|---|---|---|
| `weights` | (blocks, heads, nodes, slices) | the raw softmax memberships, float16 |
| `points` | (nodes, 3) | predicted position at that frame |
| `travel` | (nodes,) | vertical displacement from the initial blank |
| `node_type` | (nodes,) | tool nodes are marked with `OBSTACLE_NODE_TYPE` |
| `is_die` | (nodes,) bool | convenience mask for the same thing |
| `cells` | | tetrahedra, for surface extraction |
| `meta` | json | checkpoint, snapshot, block names, per-head temperatures, frame index, row-sum check |

Every block and head is stored unaveraged, so a new figure never requires a
second export. float16 is sufficient: the smallest meaningful value is the
equal-membership level `1 / n_slices`.

## Checks built into `export`

`export` aborts rather than writing questionable output.

- **Memberships must sum to one** at every node, for every block and head. If
  they do not, the hook is not reading the model's softmax. Note that this check
  passes on head-averaged data, since an average of softmaxes still sums to one,
  so it does not by itself detect that problem.
- **The frame index must not be clamped.** The rollout stores every `STRIDE`-th
  simulation step, so 1500 steps become 500 stored frames and frame `i` holds
  step `STRIDE * (i + 1)`. Passing a raw timestep as a frame index silently
  clamps the later stages to the last stored frame and produces identical panels
  for every requested time. The frame actually loaded is compared with the frame
  requested.
- A stage with no readable snapshot is reported and skipped rather than
  substituted.

## Notes on interpretation

- A figure shows one head selected from many. Heads differ greatly, and many do
  not produce a physically aligned partition at all. `select` prints the
  distribution over all candidates, which is the context for the one displayed.
- The stage models are trained separately and need not have the same depth, so a
  block index is not comparable across stages. `select` prints the block and head
  count it found for each.
- `pick_snapshot` prefers a best-validation snapshot and falls back to the
  periodic, plain and final snapshots in that order. The snapshot actually used is
  recorded in `meta` and printed, since a stage without a best-validation
  snapshot is not directly comparable with one that has it.
- `FLAT_MM` is a convention for separating stationary from moving material. The
  boundary is graded rather than sharp, so a few slices will contain nodes on
  both sides of it. Eta squared is reported alongside because it needs no such
  cut.

## Configuration

At the top of the file:

| name | meaning |
|---|---|
| `TD`, `REPO` | training data root and the repository holding `models/transolver.py`. Also settable through `SLICE_TD` and `SLICE_REPO`, or on the command line. |
| `STAGE_CKPT` | stage to checkpoint directory, relative to `TD` |
| `GEOM_PKL` | the rollout file the figures are taken from |
| `TIMES` | the timesteps sampled in each stage |
| `DT`, `STRIDE`, `STAGE_STEPS` | timing: seconds per step, rollout storage stride, and which steps belong to which stage |
| `FLAT_MM` | travel below which material counts as stationary |
| `MIN_RANGE_MM` | travel spread below which a frame has no physics to explain and is excluded from scoring |
| `VISIBLE` | membership above which a panel counts as visible |
| `FOCAL`, `CAMPOS`, `ZOOM`, `PANEL` | fixed camera, so panels are comparable |

## Requirements

| subcommand | needs |
|---|---|
| `export` | python 3.10+, torch, numpy, and the training repository importable so `models.transolver` resolves |
| `select` | numpy |
| `render`, `compare` | numpy, pyvista, matplotlib, pillow |

`render` writes per-panel intermediates to a local scratch directory, `--tmp`,
and only the assembled grids to `--out`.

## A rendering detail that matters

Every panel is rendered with the same fixed camera, and all panels of a figure
are then cropped to a single shared bounding box. Cropping each panel to its own
box and letting the layout stretch it would make the part appear a different size
in every panel, which destroys any comparison between them and gives the figure a
misleading appearance. For the same reason the camera is never refitted per panel.
