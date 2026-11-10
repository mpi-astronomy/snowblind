# Plan: snowball flagging (SnowblindStep)

Snowballs are large cosmic-ray events in the near-infrared detectors: a
saturated core inside a halo that the jump step flags only in part. This
plan covers `SnowblindStep`, which grows the jump-flagged halos and the
saturated cores, and its comparison with the jwst jump step's own snowball
flagging (`expand_large_events`, described in the STScI technical report
JWST-STScI-008545, Regan 2024). GitHub issues #5, #16 and #25.

The entries in section 3 up to 2026-11-08 were moved here on 2026-11-08 from
`PLAN-1f-cleaning-and-reset-anchored-rampfit.md`, where they were written.
Facts about the jwst pipeline are for jwst 3.0.0 and stcal 1.20.0.

## 1. Status, 2026-11-09

Done on branch `snowballs-near-edges` (committed f807f0b):

- An event needs a saturated core (issue #5): in a ramp, pixels that
  saturate in the jump group or the next; in a rate image or rateints
  cube, a hole in the jump region.
- Snowballs cut by the edge of the light-sensitive area are found and
  grown by their whole size, corners included (issue #16).
- `min_radius` defaults to 2; `snowblind_detector1` uses the default (its
  hook is `SnowblindStep(after_jumps=5)`).
- The step logs "Total snowballs = N".
- The ring dilation no longer flags pixels (0, 0), (0, 1) and (1, 0) in
  every group of a ramp.
- Provisional, to be replaced: each snowball's core (not grown) is flagged
  in every group of the next integration of a ramp, group 0 included so
  that `ResetRampFitStep` also leaves out the reset difference
  (`cores_in_next_integration`). Written before the duration was settled;
  see the decision below.

Done 2026-11-09 evening (committed d397f7c and aef5dcc; the branch was
rebased onto main after release 1.3.1 the same night: c66c1d8, 7fbb512,
caf30db; replaces the provisional rule above):

- The ring is flagged to the end of the integration (`after_jumps`
  defaults to None; set, it warns that it is deprecated and keeps the old
  group count).
- In later integrations of a ramp the core is flagged for
  `core_persistence_time` and the core grown by one pixel for
  `neighbor_persistence_time` (`flag_cores_in_later_integrations`): a group
  while the difference it ends starts within the time of the end of the
  hit's integration, group 0 included; group times from `int_times`, or
  the frame counts (`group_difference_times`). Defaults: the packaged
  `snowblind/data/snowball_persistence_<detector>.asdf` (10 NIRCam, NIS,
  NRS1, NRS2),
  read by `packaged_snowball_times`; no file: the longest packaged core and
  neighbor times (`longest_packaged_snowball_times`, 21204 s and 3814 s)
  and a warning (decided 2026-11-09, was 0 s).
- `snowblind.saturation.flag_saturation_neighbors` (function hook):
  CHARGELOSS | DO_NOT_USE on the 8 neighbors of each group's SATURATED
  pixels; IRS2 grown in the normal layout via `x_irs2`.
- `snowblind_detector1`: `n_pix_grow_sat=0` in `STEP_DEFAULTS`, the hook
  added while it is 0, `SnowblindStep(ring_width=3.0)`.
- Scripts in `scripts/`: `measure_snowball_persistence.py` (saturation
  pinned to `n_pix_grow_sat=1`), `snowball_persistence_excess.py`,
  `fit_snowball_persistence.py`, `package_snowball_persistence.py`,
  `make_snowball_persistence_figure.py`. Docs (new subsection and figure
  `docs/figures/snowball_persistence.png`), README, CHANGES, tests
  (530 pass).
- NRS1 and NRS2 files added 2026-11-09 night, uncommitted (entry of that
  date): NRS1 lengthened by its 2026 check (`package_snowball_persistence.py
  --later`); figure, docs, CHANGES and the per-detector test updated.

Decided 2026-11-08 (James):

- The dilated core (the ring of charge leaking from the saturated core) is
  flagged from the hit to the end of the integration: a simple rule, few
  pixels, and long enough on every detector measured (on NIRISS the excess
  lasts the whole integration). This replaces `after_jumps`.
- The core persists after the reset like the persistence measured for
  `PersistenceFlagStep`, but from an instantaneous injection of a large
  charge, so its decay time differs from the saturated stars'. It is to be
  flagged for a time, not a number of integrations (integration lengths
  vary widely with the readout pattern), measured for every detector and
  chosen conservatively: cover the longest decays, including any variation
  across a detector.

Decided 2026-11-08 evening, confirmed 2026-11-09 (James):

- `after_jumps` stays for one release with a deprecation warning and
  defaults to None: the ring is then flagged to the end of the
  integration; a value set by the user keeps the old group count.
- The core persistence time applies to the later integrations of the same
  exposure (the step sees one exposure). The times are packaged with
  snowblind as a small .asdf reference file per detector.
- The conservative choice is made from the distribution of decay times
  per detector, and the data for it are kept.

Decided 2026-11-09 (James):

- `snowblind_detector1` runs `SaturationStep` with `n_pix_grow_sat=0`, so
  SATURATED marks only the pixels that reached the saturation threshold.
  A snowball's core is then its un-dilated saturated pixels, flagged for
  the measured per-detector time (the 2026-11-09 times are of exactly
  these pixels). The pixels around the core are the ring, flagged to the
  end of the integration.
- `SnowblindStep` keeps its defaults. The `ring_width` documentation says
  the ring is drawn around the SATURATED flags, so how far it reaches
  from the saturated pixels depends on the `n_pix_grow_sat` that
  `SaturationStep` ran with (jwst default 1).
- Better still would be for `SaturationStep` to mark its grown pixels with
  a flag other than SATURATED, separating a saturated pixel (which
  persists) from a neighbor flagged for the charge leaking out of it.

Decided 2026-11-09 evening (James):

- `ring_width=3` in `snowblind_detector1`.
- The 99th percentile of the per-core decay times, per detector.
- The pixels 1 px from a snowball's saturated pixels get their own time:
  in the later integrations of the exposure the saturated pixels are
  flagged for the first time and the core grown by 1 px for the second
  (both in the per-detector .asdf file).
- A function post-hook of the saturation step in `snowblind_detector1`
  grows each group's saturated pixels by 1 px with CHARGELOSS and
  DO_NOT_USE in place of SATURATED.
- NIRSpec (NRS1, NRS2) to be measured the same way. Data chosen
  2026-11-09 from a MAST search of every full-frame NIRSpec dark: the
  dark monitors (1484, 4455, 6633, 9273, 12550) take one 2932 s
  integration per visit, hours to days apart, so a hit's core stays
  saturated to the end of its exposure and nothing follows it soon after.
  Main set: commissioning program 1121, visits 136-140 on 2022-03-23, 9
  IRS2 exposures back to back (gaps 58-263 s, 23.5 ks). Check: program
  9273 visits 86-89 on 2026-01-27, 4 exposures with gaps of 1.6 ks, whose
  stack shows whether the 2022 decay holds now from 1.6 to 13 ks (no data
  in a core's first 1000 s, so no per-core times from it). File lists in
  results/2026-11-08_snowball_persistence/cluster/.

## 2. Open items

1. Done 2026-11-09 for all thirteen detectors (NIRCam x10, NIRISS,
   NIRSpec x2; entries of that date). James chose the 99th percentile of
   the per-core times (darks are the strict case: with sky, the excess
   sinks into the sky noise sooner). The task was to
   measure the decay of the core's excess against the time since the end of
   the hit's integration, to the 0.005 DN/s threshold of the
   `PersistenceFlagStep` times, and pick a conservative time. Data: program
   9022 has all ten NIRCam detectors with 5 exposures back to back (2.0,
   2.6, 5.3, 7.4, 7.4 ks; gaps of about 130 s; 3.5 GB per detector),
   reaching about 25 ks after a hit; NIRISS 9279 exposures are 5
   integrations (6.4 ks) each, separate by days, so about 5 ks after a hit.
   Program 8758 exposures are days apart and mostly short.
2. Width of the dilated core in the first group after the hit: the excess
   beyond `ring_width` 2 grows with the core's size (2026-11-08 entry).
3. Bright stars whose core saturates partway up the ramp can pass as
   snowballs (the report lists this for the jwst step too). A snowball's
   saturated area hardly grows after the group following the hit; a
   star's grows group by group. Untested: the darks have no stars.
4. A hit split between two groups (NFRAMES > 1) is counted twice in the
   step's total and in EXTNCRS.
5. No cap on the size of a grown event (the jwst step caps any expansion
   at a radius of 200 pixels); the step only warns above 1000 pixels.
6. Events centered past the edge are grown from the part on the detector,
   a lower bound; a circle fitted to the arc was not tried.
7. With `n_pix_grow_sat=0` in `snowblind_detector1` (decided 2026-11-09):
   `ring_width=3` there, keeping today's reach of 3 px from the saturated
   pixels (agreed). The neighbors of saturated stars, which no step then
   flags: a function post-hook of the saturation step growing each group's
   saturated pixels by 1 px with CHARGELOSS and DO_NOT_USE in place of
   SATURATED (James: probably right, not decided). `PersistenceFlagStep`
   reads SATURATED from the rate DQ and would then flag only the saturated
   pixels of earlier exposures; the pixels 1 px from a saturated pixel do
   persist (2026-11-09 later entry), so it would need the hook's flag too.
   Done 2026-11-10 (James): CHARGELOSS kept, as jwst's flag for charge spreading from a bright pixel
   (`charge_migration`); `ResetRampFitStep` carries it into the rate and rateints DQ (stcal's DQ
   compression drops it) and `PersistenceFlagStep` reads SATURATED | CHARGELOSS, the same pixels as
   the jwst default growth its times were measured with. The jwst ramp fit (`--reset_fit=False`) still
   drops it, and recomputes the read-noise variance of CHARGELOSS groups on purpose (resample weights).
8. The pixels 1 px from a snowball's saturated pixels persist after the
   reset, for about a quarter of the saturated pixels' time (2026-11-09
   later entry); the ring is flagged only to the end of the integration.

Related, in `PLAN-1f-cleaning-and-reset-anchored-rampfit.md`: "2026-10-28:
bimodal snowball rates on the 8758 NRCA1 dark (a step at group 59)", a
common-mode step in the readout that a pixel's ramp loses when a snowball
flag splits it.

## 3. Results log

Append dated entries; do not rewrite earlier ones. Each entry: what was run
(files, settings), the numbers, and what they mean.

### 2026-10-23: snowball expansion in the jump step for the dark figures

James saw snowball rings in the NIRISS dark figure and asked whether
`expand_large_events` was on. It was: the NIRISS jump parameter
reference (0075 in the run, 0087 in context 1598, same values for
darks and images) sets expand_large_events True, min_jump_area 15,
expand_factor 1.75, sat_required_snowball True; NIRCam's sets area 5,
factor 2. The NIRISS Detector1 parameter references (0004 darks, 0005
images) override nothing. The rings are the halo beyond the expanded
disc: in the annulus 2 to 25 pixels past the flagged edge of the three
largest events the rate sits 0.01 to 0.03 DN/s above the frame.
James's choice: area 8, factor at least 2; the SnowblindStep route is
not wanted for now (the helper's `snowblind` option now also switches
the jump expansion off, since Snowblind grows the raw regions and
would otherwise grow the circles again; `jump_params` and the scripts'
`--jump key=value` override the reference). Test (scratch
`snowballs/jump_test.py`, 20 groups, OLS_C; halo = median excess rate
in the annulus 2 to 30 px beyond each flagged region of 100 px or
more; "jump px" = JUMP_DET pixels in the rate DQ, which OLS_C still
fits from the other segments, so no pixel is lost):

| dark | setting | events >= 100 px | jump px | halo median (DN/s) | halo 90th pct |
|---|---|---|---|---|---|
| NIRISS | CRDS (area 15, factor 1.75, sat) | 15 | 50496 | +0.0059 | +0.0177 |
| NIRISS | area 8, factor 2 | 17 | 53804 | +0.0055 | +0.0131 |
| NIRISS | area 8, factor 2.5 | 21 | 61757 | +0.0035 | +0.0095 |
| NIRISS | area 8, factor 2, no sat required | 18 | 62199 | +0.0051 | +0.0123 |
| NIRISS | area 8, factor 3, no sat required | 39 | 106526 | +0.0015 | +0.0069 |
| NRCA1 | CRDS (area 5, factor 2, sat) | 10 | 71231 | +0.0008 | +0.0077 |
| NRCA1 | area 8, factor 2 | 10 | 70852 | +0.0008 | +0.0077 |
| NRCA1 | area 8, factor 2.5 | 11 | 73872 | +0.0006 | +0.0062 |
| NRCA1 | area 8, factor 2, no sat required | 16 | 90874 | +0.0008 | +0.0048 |
| NRCA1 | area 8, factor 3, no sat required | 47 | 155521 | +0.0011 | +0.0037 |

The halo falls off slowly, so the expansion factor is the lever and
the area threshold barely matters at 20 groups: on NIRISS factor 2
takes the median halo from 0.0059 to 0.0055, factor 2.5 to 0.0035
(90th percentile 0.0177 -> 0.0095) for 15 percent more flagged
pixels, and only factor 3 without the saturation requirement reaches
0.0015, at twice the flagged pixels (2.6 percent of the detector).
NRCA1 halos are 5 times smaller to begin with. Chosen for the figures:
area 8, factor 2.5, saturation required, on both darks
(`--jump min_jump_area=8,expand_factor=2.5`;
`results/2026-10-23_rate_images_{nis,nrca1}`). The remaining halo at
factor 2.5 is the tail of the charge profile; a profile-following
growth is what SnowblindStep does, kept for later.

James asked for more expansion on NIRISS, factor 3. Measured with the
saturation requirement kept (the earlier factor-3 row was without it):

| dark | setting | events >= 100 px | jump px | halo median (DN/s) | halo 90th pct |
|---|---|---|---|---|---|
| NIRISS | area 8, factor 3 | 21 | 71446 | +0.0030 | +0.0080 |
| NIRISS | area 8, factor 4 | 27 | 96081 | +0.0031 | +0.0076 |

Factor 3 takes the median halo from 0.0035 (factor 2.5) to 0.0030 and
the 90th percentile from 0.0095 to 0.0080 for 16 percent more flagged
pixels; factor 4 gains nothing further at 35 percent more, so the
halo beyond 3 times the core is at the measurement floor of this
annulus statistic. NIRISS figures now use area 8, factor 3
(`results/2026-10-23_rate_images_nis_f3`); NRCA1 stays at 2.5.

### 2026-11-05: JumpPlusStep after SnowblindStep made the snowball halos noisier (issue #25)

Issue #25: snowball halos are noisier in the rate with `SnowblindStep` than
with the jump step's `expand_large_events`. Measured on CEERS F444W MEDIUM8
NRCALONG exposure 00001 (9 groups, 8 differences), one pre-jump ramp, one
stream clean, every flag set fitted on the same cleaned cube with
`ResetRampFitStep` and with the jwst `OLS_C` fit. Flag sets: A the jump step
with `expand_large_events=True` (what `snowblind_detector1
--steps.jump.expand_large_events=True` runs, with no hooks); B
`SnowblindStep(min_radius=3, after_jumps=5)` alone; C B plus `JumpPlusStep`
(the pipeline default until now); D A plus `JumpPlusStep`. "Kept" is the
median number of differences the likelihood fit uses (stcal
`determine_diffs2use`: both groups of a difference without any flag); "ERR"
is the median over the pixels of the reset fit's rate ERR divided by A's.

| pixels | count | kept A / B / C / D | ERR B / C / D |
|---|---|---|---|
| halo flagged by both methods | 7,444 | 6 / 6 / 5 / 5 | 1.00 / 1.09 / 1.09 |
| every other jump-flagged pixel | 534,592 | 6 / 6 / 5 / 5 | 1.00 / 1.09 / 1.09 |
| halo flagged by SnowblindStep only | 12,541 | 8 / 6 / 5 / 8 | 1.14 / 1.24 / 1.00 |
| SnowblindStep ring (flagged in 2+ groups) | 7,491 | 6 / 5 / 4 / 5 | 1.09 / 1.15 / 1.00 |

- SnowblindStep flags the halo in the jump group only, like the jump step;
  `JumpPlusStep`'s copy of every JUMP_DET flag into the next group is the
  whole difference where both methods flag. The OLS fit shows the same
  (+8%). The jump step's expansion covers more pixels on this exposure
  (38,462 against 28,546), so the halo area is not the cause.
- The copy removes no jump signal under the reset fit: the mean rate change
  in the halo is 0.02 ERR or less, with no excess of upward changes. Under
  OLS, leaving it out raises the share of single-jump pixels more than 1
  ERR high over the share more than 1 ERR low by up to 1.6 percentage
  points (mean +0.04 ERR): the jump left in the difference after the jump
  group is below the jump threshold.
- History: the forward copy (2023-08-11) was for OLS, where a flag on group
  N leaves out only the difference before it; stcal's jump detection has
  no frame-averaging logic, then or now. The copy of SATURATED into the
  group before (2023-08-29/30) predates stcal 1.9.0's backward pass with
  the dilution factor (2024-07, DO_NOT_USE since 2024-11), on by default
  in jwst 1.16.0 (`use_readpatt`, JP-3593, 2024-09). Of the 3,338 groups
  the copy marked here, 364 were already DO_NOT_USE, 1,512 JUMP_DET (a
  cosmic ray partway through the group; the likelihood fit drops it, OLS_C
  drops the one-group segment it leaves) and 1,462 unflagged; those look
  undamaged (the difference into them, less the median of the pixel's
  earlier differences, has the spread of unflagged pixels at the same
  group).
- Done: `JumpPlusStep` deprecated and taken out of `snowblind_detector1`,
  which also no longer sets `after_jump_flag_time1=0`: the jump step's
  flagging of groups after large jumps is the jwst default and CRDS's
  (1000 DN for 90 s on NIRISS, which is two groups of a NIS readout; off
  for NIRCam).
- To revisit: whether that flagging makes sense. It flags JUMP_DET on the
  next floor(time / group time) groups after a jump above the DN threshold,
  within the integration only. Test on a NIRISS exposure whether the
  differences after jumps above 1000 DN sit above the pixel's median
  difference more than after smaller jumps, and what the extra groups cost
  under the reset fit (each one more difference).
- Scripts and flags: results/2026-11-05_jump_plus_halos/ (`run_variants.py`
  rebuilds the ramp and the fits, about 10 minutes).

### 2026-11-05: snowballs need a saturated core, and snowballs at the edges (issues #5 and #16)

Data, all through the jump step with the `snowblind_detector1` settings and
`expand_large_events=False`: a NIRCam short-wavelength dark (program 8758,
NRCA1, RAPID, integration 1, 150 groups), CEERS F444W MEDIUM8 NRCALONG
exposure 00001 (9 groups, the flags of the issue #25 entry) and a NIRISS
dark (program 9279, NISRAPID, integration 1, first 100 groups). A "core"
is the pixels that saturate in the jump group or the next; an "event" is a
part of a filled jump region that passes the size test (a disk of
`min_radius`). "Current" is `SnowblindStep` on main, which needs no core.

Cores sit inside the halos: in the dark, 28 of the 30 cores with 20 or
more JUMP_DET pixels touching them lie in a hole of the jump mask (the
other 2 are one snowball cut by the reference pixels); in CEERS 45 of 83,
and the 38 others are cosmic-ray tracks ending in a saturated pixel, or
one or two saturated pixels with a partial ring of jumps.

| data | min_radius | current events | new events | new, at the edge | new, at the edge without a core |
|---|---|---|---|---|---|
| NRCA1 dark | 2 | 557 | 47 | 1 | 0 |
| | 3 | 34 | 35 | 1 | 0 |
| | 4 | 23 | 24 | 1 | 0 |
| CEERS | 2 | 4,805 | 163 | 1 | 1 |
| | 3 | 59 | 53 | 0 | 0 |
| | 4 | 25 | 25 | 0 | 0 |
| NIRISS dark | 2 | - | 80 | 0 | 0 |
| | 3 | 68 | 59 | 0 | 0 |
| | 4 | 35 | 27 | 0 | 0 |

- What the core requirement drops: at `min_radius` 2, mostly small
  cosmic-ray blobs and tracks; at 3, tracks and round blobs without
  saturation (CEERS, 6) and, on the NIRISS dark, the one-pixel ring of
  jumps around a core saturated in an earlier group, in each of the 8
  groups after the hit (the current code grows it in every group; the core
  ring, `ring_width` for `after_jumps` groups, covers the first ones).
  Every pixel-group the new code flags on the two darks the current code
  flags too, apart from the edge snowball.
- The edge snowball of the NRCA1 dark (group 40, centered on the first
  light-sensitive row, two cores in view) is missed by the current code at
  `min_radius` 3 and 4; the new code grows it with radius 5.2.
- Edge events without a core in view: when they needed only to be centered
  next to the edge (as the jump step's `edge_size` rule needs no
  saturation), the three datasets gave 4 / 22 / 0 at `min_radius` 2 and
  0 / 2 / 0 at 3, all small blobs or track ends touching the edge. Needing
  also a disk of `min_radius` that does not hang over the edge leaves the
  1 above (a ragged blob in CEERS at `min_radius` 2).
- `min_radius` 2 adds, over 3, 110 events on CEERS and 12 on the dark, of
  median radius 2.6 and 2.8: one saturated pixel (3x3 after the saturation
  step's growth) in a ring of jumps; on CEERS most saturate in the next
  group, a large hit late in a group of 8 frames. The jump step's
  `min_jump_area=15` is the area of a disk of radius 2.2.
- Rate mode, on a rate-style DQ built from the dark's group flags (JUMP_DET
  in any group, 10.7% of the pixels): current / new 1,356 / 242 events at
  `min_radius` 2, 42 / 43 at 3, 23 / 24 at 4. Holes stand in for cores less
  well with 150 groups of hits merged.
- Ideal disks (a halo of radius R, a core of R/3, the edge 4 reference rows
  in), share of the mask an uncut event gets that the cut one gets, at
  `min_radius` 3, center d pixels from the edge (negative: past it): R=10,
  d = -6 / -3 / 0 / 3 / 6: new 0.30 / 0.67 / 1.00 / 1.00 / 1.00, current
  0 / 0.37 / 0.58 / 0.65 / 0.94; R=20, d = 0: 1.00 against 0.61. In a
  corner with the center 6 pixels in from both edges, R=20: radius 19.5,
  share 0.97 (current 0.58). Centers past the edge give a lower bound: the
  completion uses a disk centered on the detector.
- Time for `dilate_large_area_jumps`, 150 full-frame groups of the NRCA1
  dark: current 41 s (`min_radius` 3) and 60 s (4), new 13 s for both
  (each region opened and grown in a window around it); NIRISS 100 groups:
  35 s and 45 s against 9 s.
- Defaults: James chose `min_radius` 2 (it was 4 in the step and 3 in
  `snowblind_detector1`), now the step default, which `snowblind_detector1`
  uses (its hook is `SnowblindStep(after_jumps=5)`).
- Open: an extrapolation for events centered past the edge (a circle
  fitted to the arc) was not tried, as without a core it would also turn
  cosmic-ray tracks lying along the edge into large masks.
- Scripts and flags: results/2026-11-05_snowballs_near_edges/ (`proto.py`
  is the method as a function, `synthetic.py` and `corner.py` the ideal
  disks, `evaluate.py` and `edge_events.py` the counts and cutouts,
  `check_step.py` the step against `proto.py`, `timing.py` the times;
  `run_dark_jump.py` makes the dark flags).

### 2026-11-08: snowball halos and cores in NIRCam darks, against the jwst snowball flagging

Prompted by the STScI report on the jump step's snowball flagging
(JWST-STScI-008545, Regan 2024). Its claims tested here: the charge
spilling from the saturated core into a ring around it shows "a positive
signal for the rest of the integration" (its NIRSpec example), so the jump
step flags the core grown by `sat_expand` as SATURATED to the end of the
integration, and the core again in the next integration (JUMP_DET, up to
4000 s); the halo is covered by doubling the jump region's minor axis.

Data: program 8758 darks, NRCA1 to NRCA4, RAPID, 2 integrations of 150
groups each, 10.737 s per group; Detector1 to the jump input (SIRS refpix),
the jump step with the `snowblind_detector1` settings (7 sigma) and its own
expansion off. Events: `SnowblindStep` with `growth_factor=0` and the
default `min_radius` 2, hit in group 11 or later: 321 events in the 8
integrations, equal-area radius b median 4.2 (2.4 to 28.4). The core
(pixels saturating in the jump group or the next, including the saturation
step's one-pixel growth) has equal-area radius under 2 px in 194 events
(one saturated pixel grown to 3x3), 2 to 3.5 px in 119, over 3.5 in 8.

"Excess" is a group difference minus the pixel's own mean difference
before the hit, minus the median of the same, along that row of the
cutout, over pixels far from the event (removes the 1/f striping). Pixels
flagged JUMP_DET or SATURATED in that group are left out, so every excess
below is in pixels the jump step did not flag. Errors are standard errors
of the mean, with pixel-groups taken as independent; the control bin
(15-30 px from the core) shows the method's systematic level, up to about
0.1 DN per group (0.3 in one cell of the 8 largest cores).

- Halo, jump group, mean excess per unflagged pixel by distance beyond the
  event edge: 36 DN at 0-0.25 b, 8.3 DN at 0.75-1 b, 4.8 at 1-1.5 b, 2.5
  at 1.5-2 b, 1.1 at 2-3 b, 0.4 at 3-4 b. The profile in units of b is the
  same for b <= 4 and b > 4 (for example 4.2 and 4.9 DN at 1-1.5 b), so
  growing by a multiple of b is right. `SnowblindStep` (growth_factor 2,
  mask out to 2 b beyond the edge) covers 0.90 of the excess out to 3 b;
  on NRCA1 integration 1 the jwst expansion of the same jump flags covers
  0.65, with a median mask of 221 pixels per event against 536.
- Ring around the core, excess per group after the hit, pixel next to the
  core (1 px from it), cores under 2 px / 2-3.5 px / over 3.5 px: first
  group after the hit 9.3 / 12.8 / 16.4 DN, second 0.8 / 2.1 / 2.0, third
  0.4 / 1.0 / 1.7, groups 4-5 0.19 / 0.29 / 0.17 (+-0.09 / 0.12 / 0.30),
  groups 6-10 0.12 / 0.06 / 0.38 (+-0.06 / 0.08 / 0.19); from group 11 on
  no positive excess (at most 0.06 +- 0.06). For 2-3.5 px cores the pixel
  next to the core runs 0.10-0.22 DN per group low from group 21 to the
  end (+-0.04 to 0.07), near the method's systematic level. At 2 px from
  the core the excess is 3.5 / 7.8 / 12.9 DN in the first group, 0.1 /
  0.7 / 1.1 in the second, and at most 0.4 DN after.
- No positive signal for the rest of the integration at the 0.1 DN per
  group level in these darks: the excess next to the core is gone within
  about 10 groups (110 s). `after_jumps` 2 leaves 0.4-1.7 DN in the third
  group; 5 (`snowblind_detector1`) leaves about 0.1-0.4 DN per group in
  groups 6-10.
- Width in the first group after the hit only: unflagged excess beyond
  `ring_width` 2 grows with the core, at 3 px 1.8 / 3.9 / 8.0 DN, 4-5 px
  0.5 / 1.4 / 4.4, 6-8 px 0.05 / 0.6 / 1.9; from the second group on, at
  most 0.7 DN beyond 2 px (largest cores; 0.4 for the others).
- Next integration (NRCA1, cores of integration 1, ramp of integration 2):
  the pixels that saturated show an excess falling about as 1/time, 5.3 DN
  in the first group, 2.9 per group in groups 2-3, 1.0 in 4-10, 0.36 in
  11-30, 0.15 in 31-80, 0.09 +- 0.04 in 81-149; their neighbors 1-5 px out
  show none (within +-0.1). What carries over is persistence of the
  saturated pixels, not the ring.
- Core growth after the hit (saturation spreading after the group after the
  hit, within 10 px of the core): 0 of 47 events on NRCA1, 7 of 80 on the
  NIRISS dark, 5 of 145 on CEERS; the report's NIRSpec example (its
  figures 5 and 6) shows the core growing over two groups.
- Split hits: on CEERS (MEDIUM8, 8 frames per group) 17 of 163 events are
  the second half of a hit split between two groups. Both halves are
  grown, which is right, but they are counted twice in the step's total and
  in EXTNCRS (the jump step counts them twice too). None on the RAPID darks.
- Scripts and saved rows: results/2026-11-08_snowball_report_checks/
  (`snowball_measure.py` the measurements, `run_darks.py` runs them on the
  saved flags or an uncal integration and combines them, `run_sw_darks.sh`
  the seven uncal runs, `combined_sw.txt` the combined tables,
  `next_integration.py`, `repeats_and_core_growth.py`).

### 2026-11-08 (later): the dilated core and the next integration on NRCALONG and NIRISS

Same measurement as the 2026-11-08 entry above, on program 8758 NRCALONG
(RAPID, 2 integrations of 150 groups) and the two program 9279 NIRISS darks
(NISRAPID, 5 integrations of 120 groups each), every integration through
Detector1 to the jump step with the `snowblind_detector1` settings and its
own expansion off. 123 events on NRCALONG, 870 on NIRISS (hit in group 11
or later); core radius under 2 / 2-3.5 / over 3.5 px: 61 / 50 / 12 and
475 / 333 / 62. New here: each integration's cores are also measured in
the later integrations of the same exposure (excess against the pixel's
mean difference before the hit, in the integration of the hit).

On NRCALONG the first run's control bin (15-30 px from the core) drifted
to -0.2 to -0.44 DN per group: means over a few hot pixels with changing
rates. The rerun leaves out pixels flagged in the bad-pixel mask and pixels
whose mean difference before the hit is more than 5 robust sigma from the
cutout's median; its control bin is within 0.03 DN per group (one cell
0.12). NRCALONG numbers below are from the rerun. The same cuts on NIRISS
exposure 1 change nothing beyond the errors, so the NIRISS numbers are from
both exposures without them (control within 0.07, one cell 0.16 +- 0.11).

- Pixel next to the core (1 px), excess per group after the hit, cores
  under 2 / 2-3.5 / over 3.5 px; groups 1, 6-10, 11-20, 21-40, 41-80, 81
  on:
  - NRCALONG: 6.5 / 9.3 / 14.4 in group 1; 6-10: 0.32 / 0.46 / 0.78;
    11-20: 0.13 / 0.27 / 0.64; 21-40: 0.13 / 0.17 / 0.29; 41-80: 0.07 /
    0.11 / 0.14 (+-0.06 / 0.05 / 0.11); 81 on: 0.12 / 0.13 / 0.22
    (+-0.08 / 0.07 / 0.16).
  - NIRISS: 5.6 / 13.2 / 23.4 in group 1; 6-10: 0.28 / 0.92 / 1.82;
    11-20: 0.20 / 0.59 / 1.81; 21-40: 0.12 / 0.34 / 0.99; 41-80: 0.10 /
    0.17 / 0.46 (+-0.03 / 0.03 / 0.05); 81 on: 0.08 / -0.08 / 0.27
    (+-0.06 / 0.06 / 0.09).
  - For comparison NIRCam short-wavelength (entry above): nothing above
    0.06 from group 11 on.
- So the time the excess next to the core lasts depends on the detector:
  about 10 groups (110 s) on NRCA1-4, about 40 groups (430 s) on NRCALONG
  (then about 0.1 DN per group to the end, each bin 1 to 2 standard
  errors), and the whole integration on
  NIRISS, falling roughly as 1/time (large cores: 1.8 DN per group at 11-20
  groups, 0.3 at 81 on). This is the report's "positive signal for the rest
  of the integration", seen here on NIRISS and not on NIRCam SW.
- 2 px from the core: in group 1 0.2 / 2.4 / 4.9 DN on NRCALONG and 0.5 /
  2.9 / 12.5 on NIRISS, less than on NIRCam SW (3.5 / 7.8 / 12.9); after
  that under 1.2 in group 2 and under 0.6 from group 3 on, except NIRISS
  large cores (2.5 in group 2, about 1 in groups 3-5, 0.3 at 11-20). The
  spill is narrower on these detectors and lasts longer.
- Halo coverage by `SnowblindStep` out to 3 b: 0.93 (NRCALONG), 0.92
  (NIRISS), against 0.90 on NIRCam SW. Just beyond its mask (2-3 b) the
  excess per unflagged pixel is 0.6 / 0.9 DN (NRCALONG / NIRISS); for
  events with b > 4, 0.7 / 1.1 DN, and 1.1 on NIRCam SW. On these two
  detectors the large events (b > 4) have 2 to 2.5 times the excess of
  small ones at the same distance in units of b; on NIRCam SW the two
  match.
- Cores in the next integration, excess per group at the core pixels,
  groups 1, 2-3, 4-10, 11-30, 31-80, 81 on:
  - NRCA1 (entry above): 5.3, 2.9, 0.96, 0.36, 0.15, 0.09.
  - NRCALONG: 5.3, 3.95, 1.44, 0.45, 0.18, 0.14 (+-0.04).
  - NIRISS: 9.6, 4.9, 1.96, 0.61, 0.18, 0.07 (+-0.02).
  - Their neighbors 1-5 px out: within about 0.25 in groups 1-3 and 0.05
    after, on both.
- Two integrations after the hit (NIRISS, about 400 cores): the cores show
  at most 0.05 +- 0.02 (groups 11-30) and within 0.2 +- 0.1 in the first
  groups; three and four integrations after, nothing above the errors.
  So flagging the cores in the next integration only (implemented the same
  day, every group from 0) covers what is there.
- Scripts and saved rows: results/2026-11-08_snowball_report_checks/
  (`run_exposure.py` runs every integration of an exposure and measures
  the cores in the later ones; `run_more_darks.sh` the first runs;
  `combined_nrcalong_nis.txt`, `persist_nrcalong_nis.txt` and
  `combined_robust.txt` the tables; `*_robust_*` files are the reruns with
  the outlier cuts).

### 2026-11-09: snowball core persistence times on the 9022 NIRCam darks

Run by James on an MPCDF astronode with `measure.py`
(results/2026-11-08_snowball_persistence/), snowblind f807f0b, jwst 3.0.0,
stcal 1.20.0. Program 9022 observation 4, all ten NIRCam detectors, five
exposures back to back: 02101 BRIGHT1 10 x 10, 02102 SHALLOW4 5 x 10,
02103 SHALLOW4 10 x 10, 02104 and 02105 MEDIUM8 7 x 10 (integrations x
groups); 39 integrations over 25.5 ks. Every integration goes through
Detector1 to the jump step with the `snowblind_detector1` settings (the
jwst snowball expansion off). Each group difference has the median of its
line segment (one amplifier, along the fast axis) subtracted, and the
differences are averaged per pixel in blocks of about 100 s, leaving out
those next to a SATURATED, JUMP_DET or DO_NOT_USE group. Cores are found
with `SnowblindStep`'s rule (pixels that saturate in the jump group or the
next); 860 to 1100 per SW detector, 2904 on NRCBLONG.

NRCALONG's series holds exposure 02101 only: its test ran before the other
four were on the cluster, and the full run skipped it because its output
existed. To rerun.

Definitions:

- Excess: a pixel's block rate minus its baseline, the median of its
  per-integration rates over the integrations that end before the hit's
  integration or start more than 10 ks after its end. Pixels with a
  pixeldq flag, fewer than 2 such integrations, or a baseline more than 5
  robust sigma from the control pixels' are left out.
- t: from the end of the hit's integration (its last read; the reset
  follows) to the block's mean time.
- Inner pixels: core pixels whose 8 neighbors are all in the core (2 px or
  more from the core's edge), the 3 x 3 erosion of the core.
  `SaturationStep` (`n_pix_grow_sat=1`, the default) adds a 1 px ring
  around each saturated area in every group, so a core's edge pixels are
  mostly that ring and its inner pixels the ones that saturated; most
  cores are 3 x 3, one saturated pixel. Checked on the CEERS NRCALONG
  (14201_00001) and NRCA3 (10201_00001) exposures by running
  `SaturationStep` with `n_pix_grow_sat` 0 and 1: in every group the grown
  flags are the 3 x 3 dilation of the ungrown ones, and eroding them with
  a 3 x 3 square (outside the array counted as flagged) gives the ungrown
  ones back exactly (702 px in 267 areas; 110 px in 47). In general the
  erosion also fills 1 px gaps in a saturated area (it is a closing).
- Stack: the mean over cores of each core's excess (count-weighted over
  its selected pixels) per time bin.
- Decay time of a core: its excess curve fitted from 100 to 16000 s with
  A (t / 1000 s)^-alpha, alpha fixed for the detector from a fit to its
  stack; the time at which that curve falls to 5 mDN/s.
- Null: for each core, as many random control pixels (10 px or more from
  every core) as it has selected pixels, given its hit integration and
  fitted the same way: the spread from noise alone.

Inner pixels (times in s; "fit" and "measured" are where the stack's fit
and the stack itself fall to 5 mDN/s, the latter interpolated between
bins):

| Detector | cores | alpha | fit | measured | per core 50% / 90% / 95% / 99% | null 95% / 99% |
|----------|------:|------:|----:|---------:|-------------------------------:|---------------:|
| NRCA1    |  942  | 1.58 | 1273 | 1304 | 1097 / 1399 / 1527 / 1771     | 484 / 707   |
| NRCA2    |  808  | 1.53 | 1452 | 1572 | 1287 / 1747 / 1918 / 2297     | 554 / 879   |
| NRCA3    | 1027  | 1.09 | 5267 | 5497 | 4769 / 7924 / 8960 / 11443    | 608 / 1234  |
| NRCA4    | 1041  | 1.68 | 1071 | 1147 | 921 / 1079 / 1156 / 1309      | 465 / 642   |
| NRCB1    |  802  | 1.62 | 1206 | 1358 | 1033 / 1286 / 1416 / 1608     | 507 / 708   |
| NRCB2    |  841  | 1.11 | 3699 | 4055 | 3207 / 5549 / 6402 / 8442     | 695 / 1092  |
| NRCB3    |  837  | 1.12 | 4120 | 4348 | 3563 / 6407 / 7110 / 9206     | 675 / 1078  |
| NRCB4    | 1028  | 1.02 | 5921 | 5965 | 4864 / 10127 / 11890 / 16691  | 694 / 1335  |
| NRCBLONG | 2649  | 1.21 | 2131 | 2229 | 1964 / 2838 / 3292 / 4258     | 655 / 1162  |

- Two kinds of SW detector: NRCA1, A2, A4 and B1 decay fast (alpha 1.5 to
  1.7, stack below 5 mDN/s at 1.1 to 1.6 ks); NRCA3, B2, B3 and B4 slowly
  (alpha 1.0 to 1.1, 4.1 to 6.0 ks, and a wide spread between cores, 99%
  at 8.4 to 16.7 ks). NRCBLONG is between (2.2 ks, 99% 4.3 ks).
- The whole-core mean (all core pixels, ring included) has about a third
  of the inner pixels' amplitude: its stack falls to 5 mDN/s at 0.5 to 2.1
  ks. The threshold applies pixel by pixel, so the inner pixels set the
  time.
- Core size changes the inner pixels' times little: cores of 1-9, 10-29
  and 30-99 px have medians within 25% of each other (NRCBLONG 1922, 1991,
  1919 s), except on NRCB4 (4099, 5769, 5441 s).
- Longer saturation, longer decay: medians for hits 0-300 / 300-600 / over
  600 s before the end of their integration are 934 / 1108 / 1190 s on
  NRCA1, 1661 / 1963 / 2175 on NRCBLONG, 4154 / 4837 / 5325 on NRCA3. The
  longest integrations here are 1.05 ks; longer ones are not covered.
- The power law falls faster than the stacks at late times: the measured
  stacks reach 5 mDN/s 1 to 10% later than the fits, so the per-core times
  are short by up to that much.
- Noise alone gives per-core times up to 0.6 to 1.3 ks (null 99%). On the
  fast detectors this is a large part of the spread between cores, so
  their upper percentiles overstate the real spread; on the slow ones it
  is small.
- Files: `out_9022/<detector>_series.npz` (the data), `<detector>.log`,
  `<detector>_report.txt` (analyze.py: stacks by core size, saturation
  time and quadrant, and the null), `<detector>_decay.txt` and
  `_decay.npz` (decay_times.py: per-core amplitudes and times),
  `test_saturation_closing*.txt`.

### 2026-11-09 (later): NRCALONG, and persistence by distance from the saturated pixels

NRCALONG rerun on the astronode with all five exposures (2312 cores).
Two changes to the analysis, applied to all ten detectors:

- A pixel's baseline now also leaves out the integrations after the hits
  of cores that have it within 2 px, not only of cores that hold it.
  Before, a pixel next to a core took its baseline over every
  integration, persisting ones included. The saturated-pixel times of the
  entry above move by under 1%.
- Pixels are grouped by their distance from the saturated pixels (the
  core's 3 x 3 erosion, see the entry above): saturated; 1 px (the core's
  edge pixels, the ring `SaturationStep` adds); 2 px and 3 px (1 and 2 px
  outside the core). The earlier "1 px / 2 px from core" shells and the
  2026-11-08 "neighbors 1-5 px out" were measured from the grown core, so
  they are 2 px and more from the saturated pixels.

The fits start at 100 s. Starting them at 300 or 500 s moves the fitted
stack crossing little (NRCB1 1207, 1275, 1286 s against 1358 measured)
and widens the null (99%: 690, 1160, 1955 s), so the per-core times are
instead scaled by each detector's ratio of measured to fitted stack
crossing where it is above 1 ("ratio" below).

Saturated pixels (s):

| Detector | alpha | stack, measured | ratio | per core 50% / 95% / 99% | 99% x ratio | null 99% |
|----------|------:|---------:|------:|-------------------------:|-----------:|---------:|
| NRCA1    | 1.58 | 1305 | 1.02 | 1100 / 1528 / 1773    | 1817  | 707  |
| NRCA2    | 1.53 | 1572 | 1.08 | 1284 / 1918 / 2296    | 2486  | 879  |
| NRCA3    | 1.09 | 5495 | 1.04 | 4768 / 8957 / 11439   | 11939 | 1234 |
| NRCA4    | 1.68 | 1147 | 1.07 | 921 / 1156 / 1310     | 1402  | 642  |
| NRCB1    | 1.62 | 1358 | 1.13 | 1033 / 1416 / 1608    | 1810  | 690  |
| NRCB2    | 1.11 | 4050 | 1.10 | 3203 / 6394 / 8430    | 9238  | 1092 |
| NRCB3    | 1.12 | 4363 | 1.06 | 3576 / 7134 / 9239    | 9753  | 1079 |
| NRCB4    | 1.02 | 5969 | 1.01 | 4869 / 11906 / 16716  | 16833 | 1336 |
| NRCALONG | 1.29 | 2504 | 1.07 | 2069 / 3173 / 3804    | 4065  | 1102 |
| NRCBLONG | 1.21 | 2232 | 1.05 | 1969 / 3298 / 4267    | 4463  | 1163 |

Pixels 1 px from a saturated pixel (s; the fits here run slightly long,
ratio 0.94 to 0.99, so no scaling):

| Detector | alpha | stack, measured | per core 50% / 95% / 99% | null 99% |
|----------|------:|---------:|-------------------------:|---------:|
| NRCA1    | 1.68 | 391  | 319 / 538 / 649     | 386 |
| NRCA2    | 1.76 | 401  | 345 / 566 / 668     | 432 |
| NRCA3    | 1.06 | 1161 | 1000 / 2156 / 3048  | 467 |
| NRCA4    | 1.87 | 313  | 256 / 433 / 533     | 360 |
| NRCB1    | 1.64 | 362  | 320 / 549 / 653     | 446 |
| NRCB2    | 1.20 | 707  | 608 / 1198 / 1479   | 484 |
| NRCB3    | 1.11 | 831  | 739 / 1594 / 2215   | 518 |
| NRCB4    | 1.01 | 1067 | 932 / 2089 / 3182   | 351 |
| NRCALONG | 1.49 | 581  | 497 / 800 / 991     | 478 |
| NRCBLONG | 1.45 | 534  | 482 / 775 / 935     | 480 |

- The pixels 1 px from a saturated pixel persist at about a fifth of the
  saturated pixels' excess (NRCA1 0-100 s: 48 against 276 mDN/s; NRCA3:
  93 against 392), far more than inter-pixel capacitance would carry, so
  they hold charge of their own from the hit. They fall to 5 mDN/s at
  0.3 to 1.2 ks (stack), 99% per core 0.5 to 3.2 ks.
- Pixels 2 and 3 px from a saturated pixel never reach 5 mDN/s: at most
  4.6 and 2.5 mDN/s (NRCA3, 0-100 s), under 2.8 and 1.5 on the fast and
  long-wavelength detectors.
- NRCALONG and NRCBLONG behave alike: saturated pixels at 2.5 and 2.2 ks
  (stack), 99% per core 4.1 and 4.5 ks after scaling.
- Figure: `out_9022/core_persistence_9022.png` (`plot_decay.py`), one
  panel per detector: the three stacks, the saturated pixels' fit, the
  threshold and the 99% times.

### 2026-11-09 (evening): NIRISS, a longer baseline window, and the times for all eleven detectors

NIRISS: program 9279, two pairs of exposures run on the astronode with
`measure.py --per-series 2` (`out_9279/nis_01`: exposures 001 and 002,
13 ks apart; `nis_02`: 004 and 005, 11 days apart); 5 integrations of 120
groups each, 1935 cores. `decay_times.py` now pools the cores of every
series file of a detector. A third pair was still running.

Two changes to the analysis, applied to all eleven detectors:

- Baseline window: a pixel's baseline now leaves out every integration
  that starts less than 100 ks after a hit near it (was 10 ks). On the
  25 ks NIRCam sequence this means baselines from before the hit only.
  With 10 ks, baselines after the hit still held persistence: on NRCB4
  the stack at 8-25 ks rose by 1 to 1.6 mDN/s with the longer window,
  while the control pixels' stack stayed within 0.3 mDN/s of zero with
  either window (no drift). The saturated pixels' times rose 3 to 7% on
  the fast detectors, 10 to 27% on the slow ones.
- Per-core fits use only cores with data in their first 1000 s. A NIRISS
  core hit in the last integration of an exposure has no data until the
  pair's other exposure, 13 ks or days later, where the power law is tens
  of times smaller: its fitted amplitude is mostly noise (before this
  rule the NIRISS null 99% was 15 ks). No NIRCam core is dropped by it.

Times, 99th percentile of the per-core decay times (s), scaled by the
measured / fitted stack crossing where above 1 (James chose the 99th
percentile):

| Detector | cores | alpha | stack, measured | saturated pixels | null 99% | 1 px: alpha | stack | 1 px from saturated |
|----------|------:|------:|---------:|------:|------:|-----:|-----:|------:|
| NRCA1    |  920 | 1.56 | 1375 | 1935  | 763  | 1.66 | 390  | 701  |
| NRCA2    |  799 | 1.53 | 1560 | 2493  | 941  | 1.77 | 393  | 670  |
| NRCA3    |  998 | 1.04 | 6274 | 14094 | 2010 | 1.00 | 1200 | 3814 |
| NRCA4    | 1014 | 1.68 | 1133 | 1409  | 712  | 1.89 | 309  | 554  |
| NRCB1    |  778 | 1.61 | 1382 | 1850  | 865  | 1.62 | 364  | 700  |
| NRCB2    |  821 | 1.06 | 4448 | 10606 | 1580 | 1.17 | 718  | 1656 |
| NRCB3    |  820 | 1.10 | 4692 | 10690 | 1677 | 1.11 | 811  | 2373 |
| NRCB4    | 1010 | 0.97 | 6793 | 21204 | 1970 | 0.98 | 1084 | 3633 |
| NRCALONG | 2057 | 1.28 | 2627 | 4438  | 1502 | 1.46 | 596  | 1062 |
| NRCBLONG | 2606 | 1.19 | 2344 | 4924  | 1895 | 1.43 | 540  | 993  |
| NIS      | 1309 | 1.16 | 3187 | 6459  | 1986 | 1.28 | 987  | 1924 |

("cores": cores with usable saturated pixels and data in their first
1000 s; "null 99%": the same for control pixels, saturated selection.)

- NIRISS sits with the long-wavelength NIRCam detectors: slope 1.16, the
  stack at 3.2 ks, 99% 6.5 ks. Its first 100 s are stronger than on any
  NIRCam detector (627 mDN/s saturated, 191 at 1 px, 8 at 2 px), and its
  pixels 2 px from a saturated pixel are the only ones above 5 mDN/s, in
  the first 100 s after the end of the integration only.
- NIRISS data cover 0 to 5 ks after a hit (the rest of the exposure) and
  13 to 19 ks (pair `nis_01`), nothing at 6 to 12 ks; its 99% time rests
  on the power law across that gap.
- Figure: `core_persistence.png` (`plot_decay.py`; it replaces
  `out_9022/core_persistence_9022.png` of the entry above, which used the
  10 ks window). The `_decay.txt` and `_decay.npz` files in `out_9022/`
  and `out_9279/` are from this run.

### 2026-11-09 (night): NIRSpec NRS1 and NRS2, and a check on 2026 darks

Main set: program 1121 (`out_1121/`), 9 IRS2 exposures back to back
(NRSIRS2RAPID, one integration of 20 or 245 groups each), run on the
astronode with `measure.py --jobs 1` (40 GB peak per process for a
245-group integration). The logs confirm `n_pix_grow_sat=1`; cores have a
median of 25 to 26 px. Fitted with the standard rules (`decay_times.py`:
baselines at least 2 clean integrations, cores with data in their first
1000 s).

| Detector | cores | alpha | stack, measured | saturated pixels | null 99% | 1 px: alpha | stack | 1 px from saturated | null 99% |
|----------|------:|------:|------:|------:|------:|-----:|----:|------:|------:|
| NRS1     | 1356 | 1.19 | 3457 | 6061 | 3201 | 1.27 | 735 | 2048 | 1396 |
| NRS2     |  997 | 1.21 | 3614 | 6741 | 2499 | 1.19 | 947 | 2748 | 2273 |

(Of 2128 and 1564 cores. NRS1's saturated time is its 99% of 5850 s
scaled by 1.036, measured over fitted stack crossing; NRS2's 1 px time is
2732 s scaled by 1.006.)

- NIRSpec sits between the long-wavelength NIRCam detectors and NIRISS.
  Pixels 2 px from a saturated pixel never reach 5 mDN/s (at most
  2.3 mDN/s).
- Each exposure is one long integration, so most cores stayed saturated
  for more than 600 s before the reset, the case that decays slowest: on
  NRS1 the 99% is 3.9 ks for cores saturated under 300 s and 5.9 ks for
  those over 600 s.
- NRS2's neighbor time is close to its null (2748 s against 2273 s), so
  partly noise: conservative, as on the fast NIRCam detectors.

Check: program 9273 (`out_9273/`, 2026-01-27), 4 exposures of 200 groups
with gaps of about 1.6 ks, so no core has data in its first 1000 s; fitted
with `--min-clean 1 --early-by 5000`. Its per-core times are not usable
(null 99% 4.3 to 4.8 ks); its stack is compared with 1121's over the six
time bins both have (1.5 to 12 ks), as the inverse-variance weighted mean
of the per-bin ratio 2026 / 2022 of the saturated pixels' stack:

- NRS1: 1.23 +- 0.07, above 1 in every bin (1.21, 1.18, 1.23, 1.75, 1.26,
  1.70; chi-square 15.5 for 6 bins against a ratio of 1), with the
  control pixels flat. Persistence on NRS1 looks about 23% stronger now.
- NRS2: 0.91 +- 0.06, consistent with 2022.
- The 1 px stack of 9273 is too noisy after 1.8 ks to compare (bins below
  zero; mean ratio 0.62 +- 0.25), so the neighbors take the saturated
  pixels' ratio.

James chose (2026-11-09) to lengthen NRS1's times by the check, as if
every core's excess were 1.23 times higher: each time times
1.23^(1/alpha), its own alpha, x1.195 for the core and x1.180 for the
neighbors. `package_snowball_persistence.py --later` does this for any
detector whose later stack is higher (a ratio below 1 changes nothing) and
saves the check under `later_check` in the file.

Packaged: NRS1 7241 s core, 2417 s neighbors; NRS2 6741 s and 2748 s.
Both fall within the NIRCam ranges, so the longest-time fallback stays
21204 s and 3814 s. The figure (`docs/figures/snowball_persistence.png`,
now from `scripts/make_snowball_persistence_figure.py`) adds a NIRSpec
row and draws the packaged times, so NRS1's lines include the check. The
repo's `fit_snowball_persistence.py` and `snowball_persistence_excess.py`
now have `--min-clean` and `--early-by`; rerun on the 9273 series they
give `decay.npz` files identical to `out_9273/`.
