"""Outlier detection in images with spizzle in place of drizzle.

`SparseOutlierDetectionStep` is jwst's outlier detection for images, with
its method unchanged: each group of exposures is drizzled onto a common
grid, the median of those images is blotted back onto each exposure, and
the pixels that differ from it by more than the thresholds are flagged.
The drizzling is `~snowblind.resample.SparseResampleImage`'s, and the blot
spizzle's bilinear one, from spizzle's pixel map.  The step follows jwst's
``jwst.outlier_detection.imaging.detect_outliers`` (checked against jwst
3.0.0) with those two substitutions.

spizzle is an optional dependency, ``pip install 'snowblind[spizzle]'``.
"""
import logging

import numpy as np
from jwst.datamodels import ModelLibrary
from jwst.outlier_detection.utils import median_with_resampling
from jwst.stpipe.utilities import record_step_status
from stdatamodels.jwst.datamodels import dqflags

from ._jwst_imports import add_asn_id, flag_resampled_model_crs_with_blot, guess_mode, save_blot as save_blot_file
from ._spizzle import require, spizzle
from .resample import SparseResampleImage
from .step import Step

__all__ = ["SparseOutlierDetectionStep", "flag_resampled_model_crs"]

log = logging.getLogger(__name__)

DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]


class SparseOutlierDetectionStep(Step):
    """Flag outliers in the DQ array of each input image, as jwst's OutlierDetectionStep does for images.

    The parameters are jwst's for images, with these differences:

    - no ``kernel``: the square kernel only;
    - no ``good_bits``: the pixels with DO_NOT_USE are left out, as by
      jwst's default;
    - no ``resample_data``: the exposures are always drizzled, which is
      what this step replaces;
    - the step does not read jwst's step parameters from CRDS;
    - ``pixmap_stepsize`` 16 and ``pixmap_order`` 3: the pixel map is
      interpolated with splines between the pixels where the WCS is
      evaluated, for the drizzling and the blot alike.

    Only images are supported: the step raises for spectra, coronagraphy
    and time series, for which jwst's step stays.  It is not a subclass of
    jwst's step, whose parameters stpipe would merge into its own.
    """

    class_alias = "sparse_outlier_detection"
    fits_keyword = "S_SPOUTL"
    fits_comment = "Outliers: snowblind.SparseOutlierDetectionStep"

    spec = """
        weight_type = option('ivm','exptime',default='ivm')
        pixfrac = float(min=0.0, max=1.0, default=1.0)  # Pixel shrinkage factor
        fillval = string(default='NAN')
        maskpt = float(default=0.7)
        snr = string(default='5.0 4.0')
        scale = string(default='1.2 0.7')
        backg = float(default=0.0)
        save_intermediate_results = boolean(default=False)
        search_output_file = boolean(default=False)
        in_memory = boolean(default=True) # in_memory flag ignored if run within the pipeline; set at pipeline level instead
        pixmap_stepsize = integer(min=1, default=16)  # Spacing in pixels of the WCS evaluations for the pixel map
        pixmap_order = integer(default=3)  # Spline order of the pixel map between them, 1 or 3
    """  # noqa: E501

    def process(self, input_data):
        """Flag outliers in the input images.

        Parameters
        ----------
        input_data : asn file or `~jwst.datamodels.library.ModelLibrary`
            The input association.

        Returns
        -------
        `~jwst.datamodels.library.ModelLibrary`
            The input images with DQ flags set for the outliers found.
        """
        require("SparseOutlierDetectionStep")
        library, enough = _prepare_outlier_inputs(self, input_data)
        if not enough:
            return library
        median_data, median_wcs = self._resampled_median(library)
        snr1, snr2 = _float_pair(self.snr)
        scale1, scale2 = _float_pair(self.scale)
        with library:
            for model in library:
                flag_resampled_model_crs(
                    model,
                    median_data,
                    median_wcs,
                    snr1,
                    snr2,
                    scale1,
                    scale2,
                    self.backg,
                    save_blot=self.save_intermediate_results,
                    make_output_path=self.make_output_path,
                    pixmap_stepsize=self.pixmap_stepsize,
                    pixmap_order=self.pixmap_order,
                )
                self.set_cal_step(model, "COMPLETE")
                library.shelve(model, modify=True)
        record_step_status(library, "outlier_detection", True)
        return library

    def _resampled_median(self, library):
        """The median of the groups drizzled onto a common grid with SparseResampleImage, and its WCS, as jwst's
        ``median_with_resampling`` makes it."""
        resamp = SparseResampleImage(
            library,
            blendheaders=False,
            weight_type=self.weight_type,
            pixfrac=self.pixfrac,
            fillval=self.fillval,
            # left out of the drizzled images and their median, as jwst's outlier detection leaves them out
            rejected=DO_NOT_USE,
            enable_ctx=False,
            enable_var=False,
            compute_err=None,
            pixmap_stepsize=self.pixmap_stepsize,
            pixmap_order=self.pixmap_order,
        )
        return median_with_resampling(
            library,
            resamp,
            self.maskpt,
            save_intermediate_results=self.save_intermediate_results,
            make_output_path=self.make_output_path,
        )


def _prepare_outlier_inputs(step, input_data):
    """The images an outlier step flags, as a ModelLibrary, after jwst's checks, and whether there are enough.

    jwst's mode must be imaging (the pipeline's ``mode`` attribute, or the
    exposure type); the output names get the association ID; with fewer
    than two exposures the step is recorded as skipped.

    Parameters
    ----------
    step : `~snowblind.step.Step`
        The outlier step, with ``in_memory``.
    input_data : asn file or `~jwst.datamodels.library.ModelLibrary`
        Its input.

    Returns
    -------
    library : `~jwst.datamodels.library.ModelLibrary`
    enough : bool
        False when there are fewer than two exposures.
    """
    mode = guess_mode(step, input_data)
    if mode != "imaging":
        raise ValueError(f"{type(step).__name__} supports images only, not mode {mode!r}")
    library = step.prepare_output(input_data, open_models=False)
    if not isinstance(library, ModelLibrary):
        library = ModelLibrary(library, on_disk=not step.in_memory)
    log.info(f"Outlier Detection mode: {mode}")
    add_asn_id(step, input_data)
    if len(library) < 2:
        log.warning(f"Input only contains {len(library)} exposures; outlier detection is skipped")
        record_step_status(library, "outlier_detection", False)
        return library, False
    return library, True


def _float_pair(text):
    """The two numbers of a parameter such as ``snr = '5.0 4.0'``."""
    first, second = (float(value) for value in text.split())
    return first, second


def flag_resampled_model_crs(
    input_model,
    median_data,
    median_wcs,
    snr1,
    snr2,
    scale1,
    scale2,
    backg,
    save_blot=False,
    make_output_path=None,
    pixmap_stepsize=16,
    pixmap_order=3,
):
    """Flag outliers in one image against the median blotted onto it with spizzle, updating its DQ array in place.

    jwst's ``flag_resampled_model_crs`` with spizzle's pixel map and blot in
    place of stcal's ``gwcs_blot``; the flagging is jwst's.

    Parameters
    ----------
    input_model : `~stdatamodels.jwst.datamodels.ImageModel`
        The image.
    median_data : ndarray
        The median of the drizzled images.
    median_wcs : gwcs.wcs.WCS
        The WCS of the median.
    snr1, snr2 : float
        Thresholds in noise of the first and second flagging pass.
    scale1, scale2 : float
        Factors of the blotted median's largest difference from its neighbors added to the thresholds of the two
        passes.
    backg : float
        Background added to the blotted median.
    save_blot : bool
        Save the blotted median.
    make_output_path : function
        Makes the path of the blotted median, if saved.
    pixmap_stepsize : int
        Spacing in pixels at which the WCS is evaluated for the pixel map.
    pixmap_order : {3, 1}
        Degree of the splines that interpolate the pixel map.
    """
    require("flag_resampled_model_crs")
    pixmap = spizzle.pixel_map(
        input_model.meta.wcs, median_wcs, shape=input_model.data.shape, stepsize=pixmap_stepsize, order=pixmap_order
    )
    log.info(f"Blotting {input_model.data.shape} <-- {median_data.shape}")
    blot = spizzle.blot(median_data, pixmap, out=np.empty(input_model.data.shape, dtype=np.float32))
    if save_blot:
        save_blot_file(input_model, blot, None, make_output_path)
    flag_resampled_model_crs_with_blot(input_model, blot, None, snr1, snr2, scale1, scale2, backg)
