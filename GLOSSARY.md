# snowblind

Algorithms for cleaning JWST detector data: ramp-level steps that replace or
add to steps of the jwst calibration pipeline, and the vocabulary they share
with their users.

## Language

### Snowballs

**Event**:
A contiguous region of pixels the jump step flagged in one group, before it
is known whether a cosmic ray or a snowball made it. The generic term; use
it only when that is not yet known.
_Avoid_: jump region, hit, grown region

**Snowball**:
An event that holds a core: a large cosmic-ray hit whose charge saturates
pixels and spreads into a halo around them.

**Core**:
The saturated pixels of a snowball. In a ramp, the pixels that saturate in
the jump group or the one after it; in a rate image or rateints cube, which
does not record when a pixel saturated, the pixels flagged SATURATED.

**Halo**:
The jump-flagged region around a core, which `SnowblindStep` grows to cover
the whole affected area.

**Ring**:
The core grown by a few pixels, where charge leaking from the core keeps
affecting its neighbors for the rest of the integration.

### Persistence

**Snowball persistence**:
The signal a core keeps giving in the later integrations of the same
exposure, after the reset. Flagged by `SnowblindStep`.
_Avoid_: core persistence, persistence (unqualified) for this meaning

**Persistence**:
The signal saturated pixels keep giving in later exposures. Flagged by
`PersistenceFlagStep`. Always say "snowball persistence" when the first
meaning is intended.

### Ramps and 1/f noise

**Readout stream**:
The order in which the detector reads its pixels out, along which the 1/f
noise of one group is correlated.
_Avoid_: fast axis, readout direction

**Reset-anchored fit**:
The ramp fit that anchors each pixel's slope on its reset level instead of
on the first read.
_Avoid_: reset fit

### Mosaics

**Noise factor**:
What the correlation between neighboring mosaic pixels adds to the noise of
a sum over many of them. The ERR plane of a mosaic is the root of its
summed variances times the noise factor.
