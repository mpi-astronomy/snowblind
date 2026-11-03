"""Outlier detection in images with spizzle in place of drizzle.

`SparseOutlierDetectionStep` is jwst's outlier detection for images, with
its method unchanged: each group of exposures is drizzled onto a common
grid, the median of those images is blotted back onto each exposure, and
the pixels that differ from it by more than the thresholds are flagged.
The drizzling is `~snowblind.resample.SparseResampleImage`'s, and the blot
spizzle's bilinear one, from spizzle's pixel map.  `detect_outliers` follows
jwst's ``jwst.outlier_detection.imaging.detect_outliers`` (checked against
jwst 3.0.0) with those two substitutions.

spizzle is an optional dependency, ``pip install 'snowblind[spizzle]'``.
"""
import logging

import numpy as np
from jwst.datamodels import ModelLibrary
from jwst.outlier_detection.outlier_detection_step import OutlierDetectionStep
from jwst.outlier_detection.utils import flag_model_crs, median_with_resampling, median_without_resampling
from jwst.stpipe.utilities import query_step_status, record_step_status

from ._jwst_imports import flag_resampled_model_crs_with_blot, save_blot as save_blot_file
from .resample import SparseResampleImage
from .step import Step

try:
    import spizzle
except ImportError:
    spizzle = None

__all__ = ["SparseOutlierDetectionStep", "detect_outliers", "flag_resampled_model_crs"]

log = logging.getLogger(__name__)


class SparseOutlierDetectionStep(Step):
    """Flag outliers in the DQ array of each input image, as jwst's OutlierDetectionStep does for images.

    The parameters are jwst's for images, with these differences:

    - no ``kernel``: the square kernel only;
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
        resample_data = boolean(default=True)
        good_bits = string(default="~DO_NOT_USE")  # DQ flags to allow
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
        # jwst's mode, from the pipeline's ``mode`` attribute or the exposure type, and the association ID of the
        # intermediate file names; both are private methods of jwst's step
        mode = OutlierDetectionStep._guess_mode(self, input_data)
        if mode != "imaging":
            raise ValueError(f"SparseOutlierDetectionStep supports images only, not mode {mode!r}")
        result_models = self.prepare_output(input_data, open_models=False)
        log.info(f"Outlier Detection mode: {mode}")
        OutlierDetectionStep._get_asn_id(self, input_data)

        snr1, snr2 = [float(v) for v in self.snr.split()]
        scale1, scale2 = [float(v) for v in self.scale.split()]
        result_models = detect_outliers(
            result_models,
            self.save_intermediate_results,
            self.good_bits,
            self.maskpt,
            snr1,
            snr2,
            scale1,
            scale2,
            self.backg,
            self.resample_data,
            self.weight_type,
            self.pixfrac,
            self.fillval,
            self.in_memory,
            self.make_output_path,
            self.pixmap_stepsize,
            self.pixmap_order,
        )
        if query_step_status(result_models, "outlier_detection") not in ("SKIPPED", "FAILED"):
            record_step_status(result_models, "outlier_detection", True)
            with result_models:
                for model in result_models:
                    self.set_cal_step(model, "COMPLETE")
                    result_models.shelve(model, modify=True)
        return result_models


def detect_outliers(
    input_models,
    save_intermediate_results,
    good_bits,
    maskpt,
    snr1,
    snr2,
    scale1,
    scale2,
    backg,
    resample_data,
    weight_type,
    pixfrac,
    fillval,
    in_memory,
    make_output_path,
    pixmap_stepsize=16,
    pixmap_order=3,
):
    """Flag outliers in images, as jwst's imaging outlier detection does, drizzling and blotting with spizzle.

    Parameters
    ----------
    input_models : ModelLibrary or association-type input
        The images, modified in place.
    save_intermediate_results : bool
        Save the drizzled images, the median and the blotted images.
    good_bits : str
        DQ flags of the pixels that are used.
    maskpt : float
        Fraction of the largest weight below which a pixel is left out of the median.
    snr1, snr2 : float
        Thresholds in noise of the first and second flagging pass.
    scale1, scale2 : float
        Factors of the blotted median's largest difference from its neighbors added to the thresholds of the two
        passes.
    backg : float
        Background added to the blotted median.
    resample_data : bool
        Drizzle the groups onto a common grid before the median; otherwise take it pixel by pixel.
    weight_type : {"ivm", "exptime"}
        Weighting of the exposures in the drizzling.
    pixfrac : float
        Pixel shrinkage factor of the drizzling.
    fillval : str or float
        Value of the drizzled pixels without weight, a number.
    in_memory : bool
        Keep the models in memory, if ``input_models`` is not already a library.
    make_output_path : function
        Makes the paths of the intermediate files.
    pixmap_stepsize : int
        Spacing in pixels at which the WCS is evaluated for the pixel maps.
    pixmap_order : {3, 1}
        Degree of the splines that interpolate the pixel maps.

    Returns
    -------
    ModelLibrary
        The images with outliers flagged.
    """
    if not isinstance(input_models, ModelLibrary):
        input_models = ModelLibrary(input_models, on_disk=not in_memory)

    if len(input_models) < 2:
        log.warning(f"Input only contains {len(input_models)} exposures")
        log.warning("Outlier detection will be skipped")
        record_step_status(input_models, "outlier_detection", False)
        return input_models

    if resample_data:
        resamp = SparseResampleImage(
            input_models,
            blendheaders=False,
            weight_type=weight_type,
            pixfrac=pixfrac,
            fillval=fillval,
            good_bits=good_bits,
            enable_ctx=False,
            enable_var=False,
            compute_err=None,
            pixmap_stepsize=pixmap_stepsize,
            pixmap_order=pixmap_order,
        )
        median_data, median_wcs = median_with_resampling(
            input_models,
            resamp,
            maskpt,
            save_intermediate_results=save_intermediate_results,
            make_output_path=make_output_path,
        )
    else:
        median_data, median_wcs = median_without_resampling(
            input_models,
            maskpt,
            weight_type,
            good_bits,
            save_intermediate_results=save_intermediate_results,
            make_output_path=make_output_path,
        )

    with input_models:
        for image in input_models:
            if resample_data:
                flag_resampled_model_crs(
                    image,
                    median_data,
                    median_wcs,
                    snr1,
                    snr2,
                    scale1,
                    scale2,
                    backg,
                    save_blot=save_intermediate_results,
                    make_output_path=make_output_path,
                    pixmap_stepsize=pixmap_stepsize,
                    pixmap_order=pixmap_order,
                )
            else:
                flag_model_crs(image, median_data, snr1)
            input_models.shelve(image, modify=True)

    return input_models


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
    snr1, snr2, scale1, scale2, backg : float
        As for `detect_outliers`.
    save_blot : bool
        Save the blotted median.
    make_output_path : function
        Makes the path of the blotted median, if saved.
    pixmap_stepsize : int
        Spacing in pixels at which the WCS is evaluated for the pixel map.
    pixmap_order : {3, 1}
        Degree of the splines that interpolate the pixel map.
    """
    pixmap = spizzle.pixel_map(
        input_model.meta.wcs, median_wcs, shape=input_model.data.shape, stepsize=pixmap_stepsize, order=pixmap_order
    )
    log.info(f"Blotting {input_model.data.shape} <-- {median_data.shape}")
    blot = spizzle.blot(median_data, pixmap, out=np.empty(input_model.data.shape, dtype=np.float32))
    if save_blot:
        save_blot_file(input_model, blot, None, make_output_path)
    flag_resampled_model_crs_with_blot(input_model, blot, None, snr1, snr2, scale1, scale2, backg)
