1/f cleaning along the readout stream and reset-anchored ramp fitting
=====================================================================

Draft for discussion with the jwst developers. Measurements made with jwst
e15d3e779, stcal 1.19.0 and the snowblind working tree on 2026-10-20 and 21.
All numbers come from the dated entries in
``PLAN-1f-cleaning-and-reset-anchored-rampfit.md``, section D; the scripts
under ``scripts/`` regenerate them.

Summary
-------

Two steps were built and measured on real NIRCam data:

* ``StreamCleanStep`` removes 1/f noise from group differences by filtering
  each output channel's readout-ordered sample stream with a Wiener gain
  derived from the exposure's own masked background. It reuses the scene
  mask and background machinery of ``jwst.clean_flicker_noise``.
* ``ResetRampFitStep`` runs the stcal likelihood fitter with the
  superbias-subtracted reset level prepended as an extra resultant, so that
  pixels with a single unsaturated group get a rate.

Four results matter for the pipeline:

1. Cleaning group differences and rebuilding the groups turns any part of
   the noise estimate that is not a fixed linear function of the data,
   applied identically to every difference, into a random walk along the
   ramp. The true 1/f offsets do no such thing: the offset of group j
   enters the differences j and j+1 with opposite signs, so in the sum of
   differences that makes a group only the first and last offsets survive.
   The slope fit therefore amplifies the walk about 27 times in variance
   relative to the true offsets. On a NIRCam dark the jwst median step and
   every other nonlinear variant raise the rate noise above SIRS alone. A
   strictly linear clean lowers it.
2. The Wiener filter with a cutoff at periods below 3 ms removes the
   row banding and all slower 1/f power from the group differences, lowers
   the pixel and aperture scatter of the rate image below SIRS alone at
   every scale, and with a NoiseChisel-style scene mask biases the faintest
   injected sources by 0.3 percent, the same as the row median. Without
   the cutoff it gains a further 11 percent in pixel rate scatter and loses
   12 percent of the flux of the faintest sources.
3. The reset-level scatter is 16.8 DN on NRCA1 (36 e, the size of kTC
   noise), 17.6 DN on NRCALONG and 25 DN on NIRISS. With it as a prior,
   every pixel that saturates after group 1 on a bright field gets a rate,
   and on NRCA1 the recovered cores of 30 saturated stars agree with the
   curve of growth of 223 unsaturated stars to 2 +- 1 percent.
4. The first read holds about 9 percent less signal than the slope
   predicts, plus a few DN, on NRCA1. The fitter models this as a delayed
   reset. Without the correction the reset-anchored fit shifts the median
   rate of all pixels by 1 percent.

Readout geometry
----------------

In the detector frame rows are the slow axis, the four outputs are
512-column blocks and adjacent outputs read in opposite directions. Each
output reads one pixel every 10 microseconds (the pixel time, the unit of
the stream), and 12 pixel times of overhead separate the end of one row
from the start of the next, so a row takes 524 pixel times, 5.24 ms. Times
below are given in milliseconds, with the pixel count where it helps. The
direction of channel 0 was checked on darks by comparing both directions
through the correlation of samples that straddle a row boundary at lags of
0.13 to 1 ms (13 to 100 pixel times), after removing per-row means. Channel
0 reads toward increasing detector column and the
directions alternate, on NRCA1 (2.5 to 6.7 sigma), NRCB3 (7 to 10 sigma),
NRCALONG (three channels at 2.4 to 5.6 sigma, one undetermined) and NIRISS
(3 to 11 sigma, the transposed geometry).

After the per-row mean is removed, the within-row autocorrelation of a
RAPID dark's group differences is 0.011 to 0.037 at lag 2, 0.004 to 0.007
at lags 3 to 8, 0.002 at lag 13 and zero beyond lag 100. Inter-pixel
capacitance gives 0.16 to 0.32 at lag 1. Most of the 1/f power is at
periods of a row and longer.

Noise spectra
-------------

Per channel on NRCA1 the stream spectrum of a group difference fits a power
law with slope 0.89 to 0.91 above a white level of 32 to 37 DN^2 per bin,
with the power law crossing the white level at periods of 1.0 to 1.3 ms,
and a second rise toward the shortest periods (114 to 184 at a period of
0.1 ms against 58 to 67 at 0.5 ms) that is not 1/f.

.. figure:: figures/channel_spectra.png
   :width: 100%

   What the cleaner removes, channel 0 of the NRCA1 dark. Each curve is
   the noise power in a group difference against the period along the
   readout stream, on log axes, median over the 19 differences and over
   logarithmic period bins; values below 0.1 are drawn at 0.1. Before
   cleaning (blue) the power is flat at the read-noise level of 41 DN^2 at
   periods below 0.3 ms and rises as 1/f toward long periods, by a factor
   30 at one row (5.24 ms) and a factor 1000 at a second. After the clean
   (orange) the power at every period longer than the 3 ms cutoff is gone,
   two to four orders of magnitude below the read-noise level: the row
   banding and everything slower is removed, together with the read noise's
   own share of those periods, which is what a Wiener gain near one does.
   Below the cutoff the two curves coincide by construction. Beyond 100 ms the estimate is the poorly
   measured end of a 0.66 s autocorrelation and the binning hides the
   pickup lines at 2, 4, 5 and 10 pixel periods that the raw spectrum
   shows.

Why difference cleaning can add noise
-------------------------------------

Cleaned group j is input group j minus the sum of the noise estimates of
differences 1 to j. If every estimate were a mean over the same pixels,
that sum would collapse to the mean of group j minus the mean of group 0,
as the true offsets do, and nothing would accumulate. For the per-row
median the part of the estimate that does not cancel this way is the
difference between a median and a mean, 0.25 DN per row per difference on
this dark, which predicts 0.0047 DN/s added in quadrature to the rate; the
measured change is 0.0046. A per-difference
Background2D model adds a larger such term and per-difference sigma
clipping a third, because the mask and so the filter change from one
difference to the next.

Table 1, NRCA1 dark jw08758003001, first integration, 20 groups, rate
images from OLS_C. Robust sigmas in DN/s over 98.8 percent of the pixels;
the row-mean sigma is the scatter of per-row means of the rate image, and
S is the mean stream spectrum in the band (white level 44).

=====================================  ==========  ==============  ===========  ===========  ======  ======  ======
Variant                                rate sigma  row-mean sigma  slow band S  row to 50 S  ap 9    ap 33   ap 101
=====================================  ==========  ==============  ===========  ===========  ======  ======  ======
white prediction                       0.0188      0.0008          0            0            0.0023  0.0006  0.0002
SIRS, no cleaning                      0.0205      0.0053          1204         121          0.0052  0.0037  0.0027
SIRS + jwst clean_flicker_noise median 0.0213      0.0079          403          122          0.0058  0.0046  0.0034
SIRS + row median, clipped             0.0217      0.0090          397          123          0.0061  0.0048  0.0038
SIRS + Wiener, clipped                 0.0209      0.0060          213          114          0.0059  0.0049  0.0042
SIRS + row median, linear, box 32      0.0210      0.0073          222          123          0.0045  0.0027  0.0022
SIRS + Wiener, linear, box 32          0.0197      0.0026          2.5          111          0.0039  0.0028  0.0024
SIRS + Wiener, linear, box 128         0.0196      0.0018          2.4          111          0.0034  0.0019  0.0015
SIRS + Wiener, linear, box 256         0.0195      0.0012          2.4          111          0.0031  0.0015  0.0012
SIRS + Wiener, linear, median bkg      0.0196      0.0016          7.3          112          0.0030  0.0011  0.0006
=====================================  ==========  ==============  ===========  ===========  ======  ======  ======

"Linear" means one background model per integration, fitted to the
least-squares slope of the ramp and subtracted from every difference, no
per-difference clipping and one iteration; all Wiener rows use the 3 ms
cutoff (300 pixel times, about half a row) that is now the default. The
clean with the default 32-pixel box is better than SIRS alone at every
scale: pixel scatter down 4 percent, row banding down 2 times, aperture
scatter down 25, 24 and 11 percent at 9, 33 and 101 pixels. Larger boxes
and the scalar median go further on a dark, because there is no sky to
protect.

The rows above were measured before the even-odd column correction was
added. Each amplifier of these detectors gives its even and odd columns a
slightly different offset in every read (0.02 to 0.05 DN per group on
NRCA1, with a different value and sign per amplifier, and the pattern of
one integration predicts only 8 percent of the next), so it is not a fixed
pattern that the superbias removes and it survives into the rate image as
faint vertical structure. The step now measures the offset per amplifier
and per difference from the masked science pixels and subtracts it before
the stream filter. It is one number per amplifier per difference, so it is
linear and does not random-walk. With it on, the box 32 row reads 0.0196
and 0.0023 DN/s for the pixel and row-mean scatter, the aperture scatter is
0.0038, 0.0026 and 0.0022, and the scatter of the column means of the rate
image drops from 0.0020 to 0.0013 DN/s.

What remained after that was a second effect, which the CEERS team's
striping removal (Bagley et al. 2023) also treats: an offset per column
that wanders during the integration. On two integrations of the NRCA1
dark the column means of the cleaned rate scatter 3 times more than white
noise allows, the pattern of one integration predicts only 6 percent of
the next (so it is not dark current, superbias or dark reference
residual), it is constant along the whole column within an integration
(column means from the four row quarters correlate at 0.65 to 0.80, as
high as their own noise permits), white across column index and
independent per amplifier. Per-read column jitter would leave only 0.0004
DN/s in a 20-group rate; the observed excess needs about 0.26 DN of
per-column drift over the 215 s integration. It is not a readout-stream
process, so the stream filter leaves it alone: a column offset is a
once-per-row periodic signal, white across columns, inside the 3 ms band
the filter reserves for the scene. The step now removes each column's
masked mean from each difference, per amplifier, after the background
model and before the filter (``column_offsets``, default on). CEERS
subtract a column median from the rate image once; inside the ramp a
median would random-walk, while a mean is a fixed linear operation, and
since the slope is itself a fixed linear combination of the differences
the result equals subtracting the column mean of the rate image and
cannot add noise. Row banding does not bias the column means, because it
adds the same value to every column of a row. On the dark the
column-to-column part of the column means falls from 0.0010 to 0.0002
DN/s (below the white level of 0.0004, as a mean removal should), and
what is left is the background model's structure at scales above 64
columns, which on a dark is large-scale 1/f noise the model has fitted
and the step keeps by design, just as it does for rows. On the injection
the faintest-bin bias is unchanged within its noise (-0.27 +- 2.1 to
-0.20 +- 2.4 percent) and the sky gradient is untouched. The cost on real
data is the same unmasked-wings problem as the row cleaning, spread
along a whole column, and it has not been measured on a dense field.

Where the background model comes from matters. At scales above the box
the filter removes everything from each difference and the model is added
back, so the cleaned rate's structure there is whatever the model holds. A
model fitted to the mean group difference, which is the last group minus
the first over the elapsed time, carries 3.7 times the variance of the
least-squares slope for white noise and more for 1/f; built that way the
clean was worse than SIRS at 33 to 101 pixels (0.0046 and 0.0042 against
0.0037 and 0.0027), and doubling the integrations barely helped. Fitting
the model to the slope removed the penalty. The jwst step's per-difference
model has the same weakness in a different form, one noisy estimate per
difference.

.. figure:: figures/aperture_scatter.png
   :width: 100%

   Scatter of aperture means against aperture size on the NRCA1 dark for
   SIRS alone, the jwst median step, and the linear Wiener clean with a
   32-pixel and a 256-pixel background box and with no background, with
   the white-noise prediction. With no background to protect, which is
   the right setting on a dark, the clean reaches the white-noise floor
   at every aperture size (0.0003 DN/s at 101 pixels against 0.0002
   predicted); the box model keeps whatever structure it holds above its
   scale, by design.

On NRCALONG (same dark program, same script, run with the earlier
mean-difference model) the pattern repeats in weaker form: SIRS alone
0.0199 DN/s pixel scatter and 0.0052 row-mean scatter, the jwst median
0.0202 and 0.0062, the Wiener clean with the cutoff 0.0196 and 0.0047 with
the slow band removed entirely; the large-scale numbers of that run carry
the mean-difference penalty described above and are to be regenerated.

Faint sources and the sky
-------------------------

Table 2, 1500 Gaussian sources (sigma 1.3 px, 2 to 2000 DN/s) and a sky of
0.3 + 0.2 x/2048 + 0.1 sin(2 pi y/400) DN/s added to the same dark's jump
product. Relative bias of r = 3 px aperture flux against the uncleaned fit
of the same injected data; "slope vs sky" is the fraction of the sky
structure removed.

======================================  ===========  ========  =========  ==========  =============
Variant                                 median bias  2 to 6    6 to 20    20 to 63    slope vs sky
======================================  ===========  ========  =========  ==========  =============
jwst clean_flicker_noise median         -0.0001                                       -0.126
row median, linear, box 32, clip mask   -0.0001      -0.0078   -0.0006    -0.0003     -0.088
row median, linear, box 32, chisel mask -0.0000      -0.0027   +0.0002    -0.0001     -0.076
Wiener, box 32, no cutoff, clip mask    -0.0021      -0.124    -0.0245    -0.0070     -0.090
Wiener, box 32, 3 ms cutoff, clip mask  -0.0001      -0.0107   -0.0016    -0.0005     -0.092
Wiener, box 32, 3 ms cutoff, chisel     -0.0000      -0.0026   +0.0006    +0.0002     -0.077
Wiener, box 128                         -0.0023                                       -0.733
Wiener, box 256                         -0.0022                                       -0.693
======================================  ===========  ========  =========  ==========  =============

With the chisel mask the Wiener clean and the row median are equivalent in
photometry down to the faintest injected sources, 0.3 percent at 2 to 6
DN/s and nothing above, and the sky loss drops for both because the
background model is fitted with fewer source pixels in it.

On NRCALONG the same injection gives a median bias of 0.0000 for the jwst
median and the row median and -0.0001 for the Wiener with the cutoff, with
-0.0046 at the faintest bin, and 14, 10 and 10 percent of the sky removed.

A low-pass filter removes part of any source the scene mask misses, so
the mask matters. The jwst step's mask is a per-pixel sigma clip of the
draft rate image: a pixel is a source if it sits more than 2 sigma above
the background, with no segmentation and no growth into the wings, so it
misses sources below about 20 DN/s here and the wings of every bright
source. ``StreamCleanStep`` now builds its mask along the lines of
NoiseChisel instead (Akhlaghi and Ichikawa 2015, with Gnuastro's default
parameters): the draft rate is sky-subtracted with a tiled sky model and convolved with a
2-pixel Gaussian, pixels above the median of the result are candidates
(NoiseChisel uses the 0.3 quantile; on these undersampled images the
convolved noise then percolates into clusters that survive the erosion,
so the median is used), two erosions and one opening remove what noise
alone produces, each surviving blob must exceed
the 99th percentile of the signal-to-noise of the pseudo-detections that
the same procedure finds in the sign-flipped image, and the survivors are
grown back over the eroded border and into pixels above the 0.9 quantile.
On synthetic images it masks 99.9 percent of a bright source's flux
where the clip masks the core only, still catches a source with a 3-sigma
peak, and marks 1.5 percent of a pure-noise image with a sky gradient and
a sinusoid on it.
A blanket dilation of the clip mask by 3 pixels is not a substitute: it
brings the faintest-bin bias from -1.1 to -0.4 percent but doubles the sky
structure removed, from 9 to 20 percent, because the background model
loses too many pixels.
The cutoff makes the filter blind to compact sources and costs the pixel
rate gain, which lived in the same band. Any background model smoother
than the sky structure lets the cleaner remove that structure: within one
exposure, 1/f banding and sky at periods of a few hundred rows are
degenerate and the model scale picks the compromise.

.. figure:: figures/flux_bias.png
   :width: 100%

   Relative photometric bias of injected sources against their flux, all
   with the 3 ms cutoff. With the jwst-style clip mask the Wiener clean
   loses 1 percent of the faintest sources; the NoiseChisel-style mask
   brings both the Wiener clean and the row median to 0.3 percent. The
   Wiener clean without the cutoff, not plotted, loses 12 percent there.

Reset-anchored fitting
----------------------

The pedestal of the OLS_C fit on dark integrations gives the reset scatter,
after removing the first group's own read noise:

=========  ==================================  ================  ================  ========
Detector   data                                pedestal scatter  single-read noise sigma_b
=========  ==================================  ================  ================  ========
NRCA1      9022 BRIGHT1, 10 integrations       17.6 DN           5.2 DN            16.8 DN
NRCALONG   9022 BRIGHT1, 10 integrations       18.3 DN           4.8 DN            17.6 DN
NIRISS     9279 NISRAPID, 10 integrations      25.5 DN           7.4 DN            25.0 DN
=========  ==================================  ================  ================  ========

The NIRISS value comes from two exposures of five integrations, observed
months apart and processed with different superbias references, which
the calibration expresses against one of them before combining; it
includes any change of the reset level between the exposures, which the
sky-anchored reset offset absorbs on a science exposure. One exposure
alone gives 22.0 DN. On NRCA1 the value is 36 e at a gain of 2.05, the size of kTC noise;
the per-channel medians agree to 0.5 percent, and the per-pixel pedestal mean
has a scatter of 13.9 DN of which 12.7 DN is systematic superbias error.
The prior's weight: in a BRIGHT1 5-group ramp the reset difference carries
about 1/23 of the weight of an ordinary difference.

On the program 6154 NRCA1 exposure (BRIGHT1, 5 groups), 3530 pixels
saturate from group 2 on; none gets a rate from OLS_C or LIKELY, all get
one from the reset fit. On the NRCALONG exposure of the same visit 11034 of
11036 such pixels get one. Pixels with exactly two usable differences need a
guard: with two differences the one-omit jump search is degenerate, and
without the guard it dropped the real difference.

The first read, measured as d_0 / a_rest on 242k pixels with a first
difference of 50 to 2000 DN, holds R1 = b + k a t1 with k = 0.904 and
b = -3.5 DN detector-wide on NRCA1 (per channel k = 0.897 to 0.914, b =
-2.2 to -6.0 DN) and k = 0.943, b = -2.6 DN on NRCALONG (855k pixels). The
fitter models k as a reset delayed by (1 - k) t1, which leaves
every later interval unchanged; scaling the first group's read time instead
also stretches the first ordinary interval and biased the median rate by
0.9 percent. The chi-square of the reset difference is 1.5 over all clean
pixels and 2.0 over bright ones; the all-pixel excess matches the
superbias error map, of which the dark-epoch map recovers part, and the
bright excess is not channel timing.

.. figure:: figures/first_read.png
   :width: 100%

   Ratio of the reset difference to the rate from the other differences
   against the first group difference, NRCA1, with the two-parameter model.

Curve-of-growth test on NRCA1: 30 isolated stars with cores saturating
from group 2, fully recovered by the reset fit, against the core-to-ring
ratio of 223 isolated unsaturated stars (10.22, known to 0.6 percent).
Recovered core flux over prediction minus one: +0.024 (n 22) for peaks of
2000 to 4000 DN/s, -0.066 (n 7) for 4000 to 8000, +0.016 +- 0.013
overall. On NRCALONG the field is too crowded at the long-wave PSF for a
clean reference sample and the test is inconclusive at the 5 to 10 percent
level.

The offset b must be taken from the sky pixels, which set the zero point:
with the bright-pixel regression intercept the NRCALONG median rate was
0.5 percent above LIKELY, with the sky value it is within 0.2 percent
(NRCA1 within 0.1).

The first read needs cleaning too. ``stream_clean`` works on group
differences, and for OLS_C the first read's own banding is common to
every rebuilt group and drops out of the slope; the reset-anchored fit
uses group 0 minus the prior as a difference in its own right, with
about a tenth of a group's weight, so that banding enters the rate. On
the Omega Cen NRCA1 exposure it raised the row-mean scatter of the sky
pixels from 0.0023 DN/s (OLS_C) to 0.0062, with the map prior and with
a scalar prior alike, so the map was not the source. The map does carry
row structure, 3.5 DN rms on NRCA1 against 0.3 from white noise, but
two halves of the calibration dark correlate at 0.68 in their row
means: about 3.2 DN of it is a repeatable difference between this
epoch's reset level and the superbias reference, present in every first
read as well, and only 1.5 DN is averaged-down 1/f. The step therefore
now cleans the first read by default, and when given the calibration
file it cleans it against the reset prior itself (the pedestal map with
its row, even-odd and column structure removed, plus the channel
offsets; the same map the reset step anchors to), so the filter sees
only the 1/f of the reset frame and the first read. With that the reset
fit's row-mean scatter returns to the OLS_C value (0.0024) with the
per-pixel prior intact, and bright-pixel rates agree with OLS_C to 0.1
percent. Cleaning the first read against zero with the raw map instead
leaves 0.0044, because the map's real row pattern then enters with the
opposite sign.

The map is a residual relative to a superbias. The darks that calibrate
it have had their CRDS superbias subtracted, so the map is the mean
reset level relative to that file, and it is the right prior only for a
science exposure that had the same file subtracted. The calibration now
records the superbias name the darks were processed with, the reset
step declares ``superbias`` among its reference types so that CRDS
delivers the best reference for the data, and both steps read the name
the superbias step wrote into the data. When the two names differ the
prior is corrected by the difference of the two reference images; when
the best reference differs from the one the data had subtracted the
step says so. For the Omega Cen exposures and the calibration darks the
context served the same files (NRCA1 0454, NRCALONG 0443) for
observations a year apart, so no correction applied here. The
repeatable 3 DN row pattern in the map is then the drift of the reset
level since that superbias epoch; a superbias nearer in time would
shrink it, and the bookkeeping makes that automatic.

Comparison with the JWST documentation on 1/f noise
---------------------------------------------------

The known-issues page on 1/f noise
(https://jwst-docs.stsci.edu/known-issues/1-f-noise) describes the effect
as striping along the fast-read direction from reference-voltage
fluctuations of the readout amplifiers, with a different pattern per
amplifier, and presents ``clean_flicker_noise`` as the mitigation, off by
default. Two of its figures have direct counterparts here.

Its Figure 1 shows a NIRCam dark before and after ``clean_flicker_noise``.
The two figures below are the same comparison on the NRCA1 dark of this
work, with the median reference-pixel correction (the refpix step's
own default, and the pipeline default before SIRS), SIRS (which the
current NIRCam Detector1 parameter reference selects), SIRS plus the
jwst step (median per channel), and SIRS plus ``stream_clean``. Only the
cleaning differs between the panels: every one is fitted with OLS_C over
20 groups and shown at the same stretch, and both cleaning steps run with
no background subtraction, which is the right setting for a dark. All
Detector1 steps up to and including jump run with the pipeline's CRDS
parameters, so snowball expansion in the jump step is on (266 snowballs
flagged on this 20-group dark). A
background model exists to protect the sky from the cleaning, and a dark
has none; a smooth model fitted across the whole detector also ignores
the real discontinuities in the 1/f noise at the amplifier boundaries and
hands part of that noise back (see below). The first figure is the
central 512 by 512 pixels, rows and columns 768 to 1279, which straddle
the boundary between the two middle amplifiers; the second is the full
frame. The step's ``fft`` method (NSClean) is not shown because the step
only allows it for NIRSpec exposure types and skips a NIRCam image; the
median method is what the pipeline offers NIRCam.

.. figure:: figures/dark_rate_images.png
   :width: 100%

   The central 512 by 512 pixels of the NRCA1 dark rate image, one image
   pixel per detector pixel, stretch +- 2.5 pixel sigma about the median;
   unusable pixels are drawn at the median and hot pixels clip to the top
   of the scale. The row-mean and column-mean scatter in each title are
   over this region. SIRS alone reduces the row-mean scatter of the
   median correction by a fifth. The jwst step removes the banding from
   every group difference, yet the rate image keeps as much of it as SIRS
   (0.0051 against 0.0048 DN/s): over 20 groups the non-cancelling part
   of the per-difference median accumulates into as much row structure as
   it removed, which is the random walk of section 3; with the step's
   default background model it is 0.0069. The snowblind clean leaves the
   read noise (0.0008 in row means, 0.0010 in column means); with its
   default background model it is 0.0016 and 0.0023, the difference being
   1/f noise above the box scale that the model hands back. The even-odd
   and per-column offset corrections take the column-to-column structure
   out; the dark column visible left of centre in the SIRS and jwst
   panels is a per-integration column offset, not a detector feature.

.. figure:: figures/dark_rate_images_full.jpg
   :width: 100%

   The same four rate images over the full 2048 by 2048 frame, binned 2
   by 2 for display, same stretch; the scatters in the titles are over
   the whole unbinned science area. Over the full frame the jwst step reaches
   0.0036 DN/s in row-mean scatter against 0.0047 for SIRS, so the
   central cutout is a harder region for it than the average; with its
   default per-difference background model it is 0.0047, and that model
   shows as rectangular patches, each difference's ``Background2D`` being
   a different noisy tiling of the dark that accumulates along the ramp
   like the medians. The snowblind clean with no background reaches the
   white-noise floor: 0.0004 in row means and 0.0002 in column means,
   against 0.0004 and 0.0004 expected from the pixel scatter alone, and
   0.0003 in 101-pixel block means against 0.0002. With its default
   background model the same clean gives 0.0022 and 0.0010, and smooth
   gradients remain across the frame. Those gradients are 1/f noise at
   scales above the background box: the model, fitted to this
   integration's own rate image, contains them and the step reproduces
   them. Measured in 64 by 64 pixel block means on two integrations, that
   large-scale component has 0.0024 DN/s of scatter against 0.0003 from
   white noise, correlates at 0.88 with the same component of the SIRS
   rate, where the banding hides it, and at -0.07 with the next
   integration, so it is not detector structure. On a dark the model has
   nothing to protect and should be off; on a science exposure it keeps
   the sky, and separating sky from slow 1/f needs a model with more
   information than one exposure (see Open items).

The same comparison on a NIRISS dark (program 9279, NISRAPID, first 20
of 120 groups) shows the step on the other readout geometry: on NIRISS
the fast readout axis runs along science columns, so the banding is
vertical and the amplifiers are horizontal bands.

.. figure:: figures/dark_rate_images_nis.png
   :width: 100%

   NIRISS dark, central 512 by 512 pixels, same layout and statistics as
   the NRCA1 figure. Over the full frame the column-mean scatter, which
   is the banding direction here, goes from 0.0103 DN/s (median refpix)
   and 0.0083 (SIRS) to 0.0043 with the jwst median and 0.0008 with the
   stream clean, and the pixel scatter from 0.0297 to 0.0267.

.. figure:: figures/dark_rate_images_full_nis.jpg
   :width: 100%

   The same on the full NIRISS frame, binned 2 by 2 for display. The
   rectangular patches in the first two panels are large-scale structure
   of this dark that the median reference-pixel correction and SIRS leave
   in place; with no background to protect, the stream clean removes
   them with the banding.

Its Figure 4 plots the 1/f noise in NIRCam dark data against the number
of groups in the ramp, with and without ``clean_flicker_noise``, and the
caption notes a factor of 10 improvement for small group counts and worse
results for very large group counts. Section 3 is the explanation of that
turnover: the per-difference estimate carries a non-cancelling error that
accumulates along the ramp, and the slope fit weights that accumulation
about 27 times more heavily than the true offsets, with the ratio growing
with the number of groups. For two or three groups there is little to
accumulate and the median wins by a wide margin; for twenty it loses. The
figure below repeats the measurement with the same three variants on this
dark for 3 to 100 groups.

.. figure:: figures/noise_vs_groups.png
   :width: 100%

   Row-mean scatter and pixel scatter of the OLS_C rate image against the
   number of groups fitted, NRCA1 dark, for SIRS alone, SIRS plus the jwst
   median step (its default configuration) and SIRS plus the stream clean.
   The jwst step halves the banding at 3 groups, crosses SIRS at about 12
   and is 5 times worse at 100 groups, where it also raises the pixel
   scatter by 55 percent, the turnover the documentation reports. The
   stream clean stays below SIRS by a factor 1.8 to 2.3 at every length.

A crowded field: Omega Cen
--------------------------

The same comparison on a science exposure, program 6154's Omega Cen
field (NRCA1 F115W and NRCALONG F277W, BRIGHT1, 5 groups, a different
dither from the one the reset timing was calibrated on). The jwst panel
runs ``clean_flicker_noise`` the way the program's own reduction did:
median method over full rows, flat applied before fitting, science
regions masked, scalar median background. The snowblind panels run
``stream_clean`` with the same kind of background, a scalar median, its
first read cleaned against the reset prior, and otherwise its defaults.
To be explicit about the two things the step keeps apart: the
background that is subtracted before filtering and added back is here a
single number per difference (``background_method='median'``), which
was the best choice on this field, ahead of the fitted box model and of
a scene model from the mosaic (both below); the pixels that feed the
noise fit are still chosen by the NoiseChisel-style scene mask of
section 4, which uses its own tiled sky estimate only to find sources
and never subtracts it from the data. All panels start from the same
SIRS-corrected ramp and the first three are fitted with OLS_C; the
fourth is fitted with ``reset_rampfit``. The statistics in the titles
are over the background pixels of that scene mask that lie within 5
sigma of the sky, which is where faint sources are found.

.. figure:: figures/field_rate_images_nrca1.png
   :width: 100%

   Omega Cen on NRCA1, central 512 by 512 pixels, stretch +- 2.5 sigma
   of the background pixels about the sky. Over the full frame the
   row-mean scatter goes from 0.0210 DN/s (SIRS) to 0.0076 with the jwst
   step and 0.0023 with the stream clean, and the column-mean scatter
   from 0.0286 to 0.0282 and 0.0028: the row median does nothing for the
   per-column offsets, the stream clean removes them. The stream clean
   changes the mean sky by -0.7 percent (jwst +0.4) but its pixel-level
   changes have long tails (5th and 95th percentiles -0.080 and +0.066
   DN/s against -0.028 and +0.032 for the jwst step), visible as short
   dark streaks along rows next to the brightest stars: the unmasked
   wings of saturated stars are read as slow 1/f by a linear estimator
   and subtracted along the row, where a median ignores them. With
   ``reset_rampfit`` the row-mean and column-mean scatter are 0.0024 and
   0.0028, the OLS_C values, and bright unsaturated pixels agree with
   OLS_C to 0.1 percent; its benefit is the 1646 saturated cores it fits
   and OLS_C does not.

.. figure:: figures/field_rate_images_nrca1_full.jpg
   :width: 100%

   The same on the full NRCA1 frame, binned 2 by 2 for display.

.. figure:: figures/field_rate_images_nrcalong.png
   :width: 100%

   Omega Cen on NRCALONG. Full-frame row-mean scatter 0.0863 (SIRS),
   0.0290 (jwst), 0.0063 (stream clean) and 0.0062 (stream clean with
   ``reset_rampfit``), column-mean 0.0999, 0.0999, 0.0093 and 0.0090
   DN/s. The sky changes by +1.5 percent with the jwst step and -3.3
   percent with the stream clean, the latter driven by the cluster core
   on the left of the frame, where the mask covers little and the wings
   dominate; the tails there reach -0.34 DN/s. The reset fit adds 6857
   saturated cores.

.. figure:: figures/field_rate_images_nrcalong_full.jpg
   :width: 100%

   The same on the full NRCALONG frame, binned 2 by 2 for display.

Two things were learnt on this field. First, the default background
model of ``stream_clean``, the 32-pixel-box model fitted to the slope
image, is the wrong choice for a flat-sky star field: it follows the
banding and the wings at scales above the box and the step hands them
back, so with it the stream clean reached only 0.0175 DN/s in row-mean
scatter against 0.0023 with a scalar background. The model's job is to
protect real extended structure, which the injection test has (a sky
sinusoid of 400-row period that a scalar background would remove); a
flat-sky field has none. The guidance is therefore: no background on
darks, a scalar median on flat-sky fields, the box model where there is
extended emission or a strong gradient to keep. Second, the first
version of that model, taken from the jwst helper, was a sigma-clipped
median per box that also discards any box more than 10 percent masked;
on this field it sat 15 to 20 percent below the masked mean and, because
the filter's gain is 1 at zero frequency, the step removed the
difference and lowered the sky by 16 percent on NRCA1 and 34 percent on
NRCALONG. The model is now the plain masked mean per box
(``masked_mean_background``), which is unbiased by construction; on the
injection it preserves the sky gradient five times better and on the
dark it costs 0.0004 DN/s at the 101-pixel scale.

What remains open is the streaks: a linear estimator cannot ignore
unmasked wings, and on a field this crowded the scene mask cannot cover
them all. The ``row_median`` estimator with the new ``anchor_mean``
option (which gives a median-based estimate the masked mean of the
residual, so it cannot shift the sky) has tails 40 percent shorter at
0.0035 DN/s row-mean scatter, still well below the jwst step. A scene
model from the dithered mosaic was tried and does not help on this
field (Open items): the blot's own errors are larger than the noise.

A deep field: CEERS
-------------------

The same comparison on a CEERS exposure (program 1345, NRCALONG F444W,
MEDIUM8, 9 groups of 107 s, observed 2022-06-22): a sparse field of
galaxies on a bright, smooth sky, the common JWST case. Three things
differ from the dark and Omega Cen runs. The images are compared after
flat-fielding, because the long-wave rate image before the flat carries
the detector's response, a 10 percent large-scale pattern that
dominates any row or column statistic. The step applies the flat
before measuring its background and subtracts that background times
the flat from the raw difference, since the background of a sky field
is the sky times the detector response, while the 1/f noise is additive
in raw DN and is filtered there. And the background is the scalar
median: with the flat applied the sky is smooth, and the box models
hand back the slow 1/f for no gain.

.. figure:: figures/field_rate_images_ceers_nrcalong.png
   :width: 100%

   CEERS on NRCALONG, flat-fielded, central 512 by 512 pixels, stretch
   +- 2.5 sigma of the sky pixels about the sky. Over the full frame the
   row-mean scatter of the sky pixels goes from 0.0013 DN/s (SIRS) to
   0.0006 with the jwst step and 0.0003 with the stream clean, the
   column-mean scatter from 0.0015 to 0.0015 and 0.0002: the row median
   cannot see the per-column offsets, the stream clean removes them.
   The sky changes by -0.06 percent, its large-scale structure by 0.2
   percent, and the per-pixel change lies within -0.005 and +0.003 DN/s.
   On a 9-group ramp of 107 s groups the 1/f left by SIRS is small to
   begin with, so the gain is a factor 4 to 5 rather than the factor 10
   of the darks.

.. figure:: figures/field_rate_images_ceers_nrcalong_full.jpg
   :width: 100%

   The same on the full frame, binned 2 by 2 for display.

The exposure predates the calibration darks by three years and had a
different superbias subtracted (reference 0220 against 0443, 11700 DN
apart). The reset step reports the mismatch and uses the pedestal map
as the residual against the exposure's own superbias, which is the
quantity that transfers between epochs; and it carries the first-read
timing from the calibration's 10.7 s first-group time to this
pattern's 48.3 s as a fixed delay of 0.6 s, which the two-parameter fit
on this exposure confirms (k of 1.009 against 0.987 predicted, within
its errors). With that, the reset point is used for 85 percent of
pixels, all 432 pixels saturated from group 2 get a rate, and the
median rate agrees with the likelihood fit to 0.02 percent.

This exposure also found a frame the step had not been cleaning. For
frame-averaged readouts the pipeline carries the first frame of the
first group separately, the ZEROFRAME, and the likelihood fit uses it
as a resultant of its own; BRIGHT1 has no separate zeroframe, so Omega
Cen never exercised it. Left uncleaned it carried the full 1/f banding
of a single read into the reset fit, which showed 0.0011 DN/s of row
structure against 0.0002 for OLS_C on the same cleaned ramp. The step
now cleans the zeroframe as it cleans the first read, as a difference
against the reset prior with the background scaled to one frame time,
and the reset fit matches OLS_C on this field (0.0003 and 0.0002 DN/s)
while fitting 990 more pixels.

What this builds on
-------------------

Neither step starts from scratch, and the comparisons in this write-up
are against methods that do most of the work.

* The reference-pixel correction is SIRS, Simple Improved Reference
  Subtraction (Rauscher, Fixsen and Mosby 2022, JATIS 8, 028002,
  doi:10.1117/1.JATIS.8.2.028002). Every chain here, including the jwst
  one, runs on SIRS-corrected ramps, and SIRS is now the Detector1
  default for NIRCam. ``stream_clean`` removes what SIRS leaves: the
  1/f within and between the output channels that the reference
  pixels cannot see, and the per-read even-odd and column offsets.
* The idea of cleaning rows and columns after masking the sources is
  the CEERS team's (Bagley et al. 2023, ApJL 946, L12,
  doi:10.3847/2041-8213/acbb08), who subtract masked row medians per
  amplifier and masked column medians from the rate images. The
  per-column offsets of section 3 are that method's column pass moved
  into the ramp as a mean, so that it stays linear.
* The jwst ``clean_flicker_noise`` step, the comparison throughout,
  descends from Chris Willott's image1overf for its median method and
  from NSClean for its Fourier method, which is also the closest
  relative of the Wiener clean. image1overf is software without a
  paper of its own: Willott, C., ``image1overf.py`` in
  https://github.com/chriswillott/jwst (GPL-3.0), described and
  recommended on the STScI page on NIRCam 1/f noise removal methods
  (https://jwst-docs.stsci.edu/known-issues/nircam-known-issues/nircam-1-f-noise-removal-methods).
  NSClean is Rauscher 2024, PASP 136, 015001,
  doi:10.1088/1538-3873/ad1b36.
* The ramp fit is Timothy Brandt's likelihood method (Brandt 2024,
  PASP, doi:10.1088/1538-3873/ad38d9, and the companion jump-detection
  paper doi:10.1088/1538-3873/ad38da), as implemented in stcal as the
  LIKELY algorithm. ``reset_rampfit`` only adds one resultant, the
  reset level with its own variance and timing, to that machinery.
* The scene mask follows NoiseChisel (Akhlaghi and Ichikawa 2015,
  ApJS 220, 1).


Observations on stcal and jwst
------------------------------

* ``ramp_fit_data`` scales the read-noise array it is given in place
  (``readnoise_2d *= gain_2d / sqrt(2 nframes)`` for OLS). This is a side
  effect rather than a bug in the fit: the pipeline step discards the
  array afterwards, so pipeline products are unaffected, but any caller
  that reuses the array it passed in gets electrons per single read
  instead of the reference file's DN. It cost this work one wrong
  calibration file (sigma_b 15.9 instead of 16.8 DN) before it was found;
  a copy inside ``ramp_fit_data`` would remove the trap.
* The OLS_C pedestal is ``group0 - slope * ((nframes + 1) / 2 + dropframes)
  / (nframes + groupgap)`` with the slope in DN/s and no group time; on a
  ramp with reset 20 DN, 1 DN/s and 10 s groups it is 29, not 20. This is
  a bug in the optional ``pedestal`` output only: the slope, variance and
  DQ outputs do not use it. The pedestal is wrong by the slope times the
  group time less about one second, so it is right for darks, where the
  slope is near zero, and increasingly wrong for pixels with signal. No
  downstream pipeline step reads it; the calibration here uses it on
  darks, where the error is below 0.01 DN.
* ``average_dark_current`` reaching ramp fitting is zero for NIRCam, and
  the plan's reading of the two fitters is that each adds it to a DN/s
  rate before dividing by gain once. That reading was not re-checked in
  this work. If it is right it is a units bug that has no effect on
  NIRCam, whose dark reference files carry no dark-current array, and
  that would mis-scale the dark contribution to the Poisson variance by
  the gain for an instrument whose files do carry one.
* The jwst ``clean_flicker_noise`` step inherits the random walk of
  section 3: on this dark it raises the rate noise from 0.0205 to
  0.0213 DN/s and the row-mean scatter from 0.0053 to 0.0079. This is not
  a coding error but a property of the design, a per-difference nonlinear
  estimate (the median, the per-difference background model, the
  clipping) followed by a group rebuild; it means that on a dark or a
  sparse field a user who enables the step gets a rate image that is 4
  percent noisier per pixel and 50 percent noisier in row banding than
  without it, while the group differences themselves look cleaner. A
  linear estimator applied identically to every difference avoids it, and
  the step's existing median could not be made linear.

Open items
----------

* Above the background box scale the cleaned rate reproduces the model,
  so sky structure at periods of a few hundred rows is protected only to
  the extent the model follows it; a model informed by more than one
  exposure, or by the mosaic, would let the box grow without losing sky.
* The scene mask stops where a source drops below the clipping level and
  its wings extend further; the step has a ``mask_dilation`` option to
  grow the masked regions, not yet measured. The 4-pixel reference border
  is always excluded from the fit.
* The pixel-to-pixel spread of the first-read response, which the field
  chi-square of 2 points at, needs a brighter or repeated field to measure.
* The dark and injection experiments on NRCALONG, and a NIRISS field.
* A scene model from the dithered mosaic, blotted back into each
  exposure, in place of the per-exposure background model. Tried on
  Omega Cen NRCA1 (four dithers, Image3 with tweakreg and skymatch
  skipped, the outlier-detection blots times the flat as the model, the
  step's ``scene_model`` option): it does not help there. The blot sits
  5 percent above the exposure on the faintest pixels and its error
  field on the sky is 0.014 DN/s rms in 16-pixel blocks, larger than
  the 1/f being removed; the filter's unit gain at zero frequency turns
  the mean of that error into a sky shift of 2 to 5 percent per channel
  and its structure into row and column scatter four times that of the
  scalar background, with star photometry unchanged to 0.2 percent.
  Re-anchoring the model's level per channel repairs the sky but not the
  structure; smoothing the blot before use goes the wrong way (a
  4-pixel Gaussian lifts the sky by 15 percent, since it spreads star
  light into the sky pixels), and removing the model's row-aligned part,
  which is the other dithers' residual banding carried through the
  mosaic, recovers 40 percent of the structure but not the rest. A
  four-dither drizzled median of an undersampled crowded field is not
  accurate enough at the slow scales; the route stays open
  for sparse fields and smooth extended emission, where the blot error
  is small and the box-size question is the one that matters, and a
  crowded field would need a PSF-based star model instead.
* Extended-source photometry has not been tested; the injection used
  compact Gaussians only.
