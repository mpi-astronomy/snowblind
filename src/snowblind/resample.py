"""Resampling of images with spizzle in place of drizzle.

`SparseResampleStep` is jwst's resample step for images, and
`SparseResampleImage` the class that does the work.  `SparseResampleImage` is jwst's `~jwst.resample.resample.ResampleImage`
with drizzle's square kernel replaced by spizzle's sparse operator: the
pixel map, the operator of each exposure, and the sums of the science data,
the weights and the variance planes, in float64.  The weights are
`drizzle_weights`, stcal's weights from the DQ flags each caller rejects,
and everything else, from the output grid to the metadata, is jwst's and
stcal's.

The variance planes, by ``variance``:

- ``"exact"`` (the default): the variance of each output pixel,
  ``(W * W) @ var / wsum**2`` for each plane, without stcal's division by
  the pixel scale ratio squared.  Drizzling correlates neighboring output
  pixels, which the pixel variance leaves out.  The noise factor, about the
  variance of a sum over a large aperture over the sum of the pixel
  variances in it (`spizzle.NoiseFactor`), is computed in a second pass over
  the exposures and kept as `SparseResampleImage.noise_factor`, and ERR is
  ``sqrt(sum of the variance planes * noise factor)``.
- ``"stcal"``: stcal's variance planes and ERR, from the same operators, to
  compare with the jwst step.

Only the drizzle layer is replaced.  `_SpizzleResample` overrides the
methods of stcal's `~stcal.resample.Resample` that call drizzle
(``reset_arrays``, ``add_model`` and ``finalize``) and sits between jwst's
class and stcal's in the method resolution order, so the jwst methods,
which call ``super()``, reach it.  It uses private parts of stcal's class,
checked in `snowblind._jwst_imports`.

spizzle is an optional dependency, ``pip install 'snowblind[spizzle]'``;
without it this module imports, and `SparseResampleImage` raises.
"""
import logging
from pathlib import Path

import numpy as np
from jwst.datamodels import ImageModel, ModelContainer, ModelLibrary
from jwst.lib.pipe_utils import match_nans_and_flags
from jwst.resample.resample import ResampleImage, copy_asn_info_from_library
from jwst.resample.resample_step import ResampleStep
from jwst.resample.resample_utils import load_custom_wcs
from stcal.resample import Resample
from stcal.resample.utils import get_tmeasure, is_imaging_wcs, resample_range
from stdatamodels import filetype
from stdatamodels.jwst.datamodels import dqflags

from . import _jwst_imports  # noqa: F401  (checks the stcal internals used here)
from ._spizzle import require, spizzle
from .step import Step

__all__ = ["SparseResampleStep", "SparseResampleImage", "drizzle_weights", "VARIANCE_MODES", "RESAMPLE_REJECTED"]

log = logging.getLogger(__name__)

VARIANCE_MODES = ("exact", "stcal")

DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]
NON_SCIENCE = dqflags.pixel["NON_SCIENCE"]
# the DQ flags of the pixels left out of a mosaic, as jwst's resample step leaves them out
RESAMPLE_REJECTED = DO_NOT_USE | NON_SCIENCE

# suffix of the noise factor file that SparseResampleStep saves next to the mosaic
NOISE_FACTOR_SUFFIX = "nfactor"


def drizzle_weights(dq, weight_type, rejected, var_rnoise=None, exposure_time=None):
    """The drizzle weight of each pixel of an exposure, zero where its DQ has any of the ``rejected`` flags.

    The weights of stcal's ``build_driz_weight``, with the pixels to leave
    out named by their flags.

    Parameters
    ----------
    dq : ndarray of uint32
        The exposure's DQ array.
    weight_type : {"ivm", "exptime"}
        "ivm": the inverse of ``var_rnoise``, zero where that is not finite,
        and 1 everywhere, with a warning, when ``var_rnoise`` is missing or of
        another shape.  "exptime": ``exposure_time`` everywhere.
    rejected : int
        The DQ flags of the pixels left out, such as ``DO_NOT_USE |
        NON_SCIENCE``.
    var_rnoise : ndarray or None
        The read-noise variance, for "ivm".
    exposure_time : float or None
        The measurement or exposure time, for "exptime".

    Returns
    -------
    ndarray of float32, the shape of ``dq``
    """
    if weight_type == "ivm":
        if var_rnoise is None or var_rnoise.shape != dq.shape:
            log.warning("No VAR_RNOISE of the exposure's shape: its pixels get the weight 1")
            weights = np.ones(dq.shape, dtype=np.float32)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = 1.0 / var_rnoise
            weights[~np.isfinite(weights)] = 0.0
    elif weight_type == "exptime":
        weights = np.full(dq.shape, exposure_time, dtype=np.float32)
    else:
        raise ValueError(f"weight_type must be 'ivm' or 'exptime', not {weight_type!r}")
    return np.where((dq & rejected) == 0, weights, 0.0).astype(np.float32)


class _SpizzleResample(Resample):
    """stcal's Resample with spizzle's sums in place of drizzle's.

    ``variance`` and ``rejected`` must be set before ``Resample.__init__``
    runs, as `SparseResampleImage` does.
    """

    def reset_arrays(self, n_input_models=None):
        """Start new sums for the science data, the weights and the variance planes.

        Parameters
        ----------
        n_input_models : int or None, optional
            Not used: spizzle's context grows with the exposures.
        """
        if not hasattr(self, "_output_model") or self.is_finalized():
            self._output_model = self.create_output_model()
        shape = self._output_array_shape
        self._fill = float(self.fillval)
        n_var = len(self.variance_array_names) if self._exact_variance() else 0
        self._coadd = spizzle.Coadd(shape, n_var=n_var, context=self._enable_ctx)
        self._err_coadd = spizzle.Coadd(shape, fillval=self._fill) if self._compute_err == "driz_err" else None
        # made with the first exposure, which sets the pixel scale ratio
        self._stcal_variance = None
        # made with the first exposure of the second pass, from the final weight sums
        self._noise = None
        self.init_time_counters()
        self._finalized = False

    def add_model(self, model):
        """Add one exposure to the sums.

        Parameters
        ----------
        model : dict
            The exposure, as for stcal's ``Resample.add_model``.
        """
        if self._finalized:
            raise RuntimeError("Resampling has been finalized; call reset_arrays before adding models.")
        if self._noise is not None:
            raise RuntimeError("The noise factor pass has started; no more models can be added.")
        self.validate_input_model(model)
        self._n_res_models += 1
        data = model["data"]
        if data.ndim != 2:
            raise RuntimeError(f"Input model '{model['filename']}' is not a 2D image.")

        self._compute_pixel_scale_ratio(model)
        iscale = self._get_intensity_scale(model)
        operator = self._operator(model)
        if not model["subtracted"] and model["level"] is not None:
            data = data - model["level"]

        if self._exact_variance():
            self._coadd.add(operator, data, self._variances(model), iscale)
        else:
            self._coadd.add(operator, data, iscale=iscale)
        if self._enable_var and self.variance == "stcal":
            if self._stcal_variance is None:
                self._stcal_variance = spizzle.StcalVariance(
                    self._output_array_shape,
                    n_var=len(self.variance_array_names),
                    rnoise=self.variance_array_names.index("var_rnoise"),
                    weight_type=self.weight_type,
                    pixel_scale_ratio=self._scale_ratio(),
                )
            self._stcal_variance.add(operator, self._variances(model), iscale, exposure_time=get_tmeasure(model)[0])
        if self._err_coadd is not None:
            self._err_coadd.add(operator, model["err"], iscale=iscale)

        if (group_id := model["group_id"]) not in self._group_ids:
            self.update_time(model)
            self._group_ids.append(group_id)
            self._output_model["pointings"] += 1

    def add_noise_factor(self, model):
        """Add one exposure to the noise factor, in a second pass after all are added with `add_model`.

        Parameters
        ----------
        model : dict
            The exposure, as for `add_model`.
        """
        if self._finalized:
            raise RuntimeError("Resampling has been finalized; call reset_arrays before adding models.")
        if self._noise is None:
            self._noise = spizzle.NoiseFactor(self._coadd.out_wht)
        planes = dict(zip(self.variance_array_names, self._variances(model)))
        var = np.sum([planes[name] for name in self._error_variance_names() if planes[name] is not None], axis=0)
        self._noise.add(self._operator(model), var, self._get_intensity_scale(model))

    def finalize(self):
        """Form the output arrays from the sums, as stcal's finalize does from drizzle's."""
        if self._finalized:
            return
        if self._exact_variance() and self._noise is None:
            raise RuntimeError(
                "The exact variance needs the noise factor: add every model again with add_noise_factor "
                "before finalize, as resample_group and resample_many_to_one do."
            )
        self._finalized = True
        self._output_model["pointings"] = len(self.group_ids)

        wht = self._coadd.out_wht
        data = self._coadd.out_img
        data[wht == 0] = self._fill
        self._output_model["data"] = data.astype(np.float32)
        self._output_model["wht"] = wht.astype(np.float32)
        if self._enable_ctx:
            self._output_model["con"] = self._coadd.out_ctx

        if self._enable_var:
            planes = self._coadd.out_var if self._exact_variance() else self._stcal_variance.out_var
            for name, plane in zip(self.variance_array_names, planes):
                self._output_model[name] = plane.astype(np.float32)
        if self._exact_variance():
            self._output_model["noise_factor"] = self._noise.noise_factor.astype(np.float32)

        if self._compute_err == "driz_err":
            self._output_model["err"] = (self._err_coadd.out_img / self._scale_ratio()).astype(np.float32)
        elif self._enable_var and self._compute_err == "from_var":
            var_components = [self._output_model[name] for name in self._error_variance_names()]
            var = np.nansum(var_components, axis=0)
            # nansum gives zero where all are NaN
            var[np.all(np.isnan(var_components), axis=0)] = np.nan
            if self._exact_variance():
                var = var * self._output_model["noise_factor"]
            self._output_model["err"] = np.sqrt(var).astype(np.float32)

        self._coadd = self._err_coadd = self._stcal_variance = self._noise = None
        self.finalize_time_info()

    def _exact_variance(self):
        """Whether the variance planes are the exact ones, with the noise factor."""
        return self._enable_var and self.variance == "exact"

    def _scale_ratio(self):
        """Output over input pixel scale, 1 if unknown, as stcal scales ERR and its variance by."""
        return 1.0 if self.pixel_scale_ratio is None else self.pixel_scale_ratio

    def _error_variance_names(self):
        """The variance planes that ERR is formed from."""
        return self.variance_array_names if self.error_from_variances is None else self.error_from_variances

    def _variances(self, model):
        """The exposure's variance planes in stcal's order, None for one that is missing or of another shape."""
        return [model[name] if self._check_var_array(model, name) else None for name in self.variance_array_names]

    def _operator(self, model):
        """spizzle's operator of one exposure onto the output grid, with its `drizzle_weights` and stcal's input
        range."""
        shape = model["data"].shape
        pixmap = spizzle.pixel_map(
            model["wcs"], self.output_wcs, shape=shape, stepsize=max(1, int(self.pixmap_stepsize)), order=self.pixmap_order
        )
        weights = drizzle_weights(
            model["dq"], self.weight_type, self.rejected, var_rnoise=model.get("var_rnoise"),
            exposure_time=get_tmeasure(model)[0],
        )
        xmin, xmax, ymin, ymax = resample_range(shape, model["wcs"].bounding_box)
        return spizzle.drizzle_operator(
            pixmap,
            self._output_array_shape,
            pixfrac=self.pixfrac,
            weights=weights,
            region=(slice(ymin, ymax + 1), slice(xmin, xmax + 1)),
        )


class SparseResampleImage(ResampleImage, _SpizzleResample):
    """jwst's ResampleImage, drizzling with spizzle.

    Parameters
    ----------
    input_models : `~jwst.datamodels.library.ModelLibrary`
        The exposures to resample.
    variance : {"exact", "stcal"}
        The exact variance planes with the noise factor, or stcal's variance
        planes and ERR.
    pixmap_stepsize : int
        Spacing in input pixels at which the WCS is evaluated for the pixel
        map, which is interpolated between them; 1 evaluates every pixel.
    pixmap_order : {3, 1}
        Degree of the interpolating splines of the pixel map.
    enable_ctx : bool
        Make the context, one int32 plane of the output size per 32
        exposures; off by default, unlike jwst's class.
    rejected : int
        The DQ flags of the input pixels left out (weight 0); by default
        `RESAMPLE_REJECTED`, DO_NOT_USE and NON_SCIENCE.
    on_output : callable or None
        Called with each output model once it is finalized, the mosaic or
        each group's image.
    **kwargs
        The other parameters of jwst's ``ResampleImage``.  ``kernel`` must be
        "square", ``propagate_dq`` is not supported, ``fillval`` must be a
        number or a string of one, such as "NAN", and ``good_bits`` is
        replaced by ``rejected``.
    """

    def __init__(
        self, input_models, variance="exact", pixmap_stepsize=16, pixmap_order=3, enable_ctx=False,
        rejected=RESAMPLE_REJECTED, on_output=None, **kwargs,
    ):
        require("SparseResampleImage")
        if variance not in VARIANCE_MODES:
            raise ValueError(f"variance must be one of {VARIANCE_MODES}, not {variance!r}")
        if (kernel := kwargs.get("kernel", "square")) != "square":
            raise ValueError(f"SparseResampleImage has only the square kernel, not {kernel!r}")
        if kwargs.get("propagate_dq", False):
            raise ValueError("SparseResampleImage does not propagate DQ")
        if "good_bits" in kwargs:
            raise ValueError("SparseResampleImage takes the DQ flags it leaves out as rejected, not good_bits")
        fillval = kwargs.get("fillval", "NAN")
        try:
            float(fillval)
        except (TypeError, ValueError):
            raise ValueError(f"fillval must be a number or a string of one, such as 'NAN', not {fillval!r}") from None
        self.variance = variance
        self.rejected = rejected
        self.on_output = on_output
        # the library indices of the exposures added, for the second pass of the exact variance
        self._noise_factor_indices = None
        super().__init__(
            input_models, pixmap_stepsize=pixmap_stepsize, pixmap_order=pixmap_order, enable_ctx=enable_ctx, **kwargs
        )
        if not is_imaging_wcs(self.output_wcs):
            raise ValueError("SparseResampleImage resamples images only; the output WCS is not an imaging WCS")

    @property
    def noise_factor(self):
        """ndarray of float32 or None: the noise factor of the exact variance, once finalized."""
        return self._output_model.get("noise_factor")

    def add_noise_factor(self, model):
        """Add one exposure to the noise factor, in a second pass after all are added with ``add_model``.

        Parameters
        ----------
        model : ImageModel
            A JWST data model that was added with ``add_model``.
        """
        super().add_noise_factor(
            self.input_model_to_dict(model, weight_type=self.weight_type, enable_var=True, compute_err=None)
        )

    def finalize(self):
        """Finalize the output as jwst's class does, after the second pass for the exact variance's noise factor.

        The second pass runs only with the exact variance (``enable_var`` on,
        ``variance="exact"``), over the exposures that `resample_group` or
        `resample_many_to_one` added; then ``on_output`` gets the output.
        """
        if self.is_finalized():
            return
        if self._exact_variance() and self._noise is None and self._noise_factor_indices is not None:
            log.info("Computing the noise factor in a second pass over the exposures")
            with self.input_models:
                for index in self._noise_factor_indices:
                    model = self.input_models.borrow(index)
                    self.add_noise_factor(model)
                    self.input_models.shelve(model, index, modify=False)
        self._noise_factor_indices = None
        super().finalize()
        if self.on_output is not None:
            self.on_output(self.output_jwst_model)

    def resample_group(self, indices):
        """jwst's, with the second pass for the exact variance's noise factor (see `finalize`).

        Parameters
        ----------
        indices : list of int
            The library indices of the models of one group.

        Returns
        -------
        `~stdatamodels.jwst.datamodels.ImageModel`
        """
        self._noise_factor_indices = list(indices)
        return super().resample_group(indices)

    def resample_many_to_one(self):
        """Resample and co-add all the input models to one output, as jwst's class does.

        The models are not written back to the library, being unchanged, and
        `finalize` makes the second pass for the exact variance's noise factor.

        Returns
        -------
        `~stdatamodels.jwst.datamodels.ImageModel`
            The resampled and co-added image.
        """
        log.info("Resampling science and variance data with spizzle")
        self._noise_factor_indices = range(len(self.input_models))
        with self.input_models:
            for model in self.input_models:
                self.add_model(model)
                self.input_models.shelve(model, modify=False)
        self.finalize()
        self.output_jwst_model.meta.filename = self.output_filename
        copy_asn_info_from_library(self.input_models, self.output_jwst_model)
        return self.output_jwst_model


class SparseResampleStep(Step):
    """Resample images onto a regular grid with spizzle's square kernel.

    jwst's ``ResampleStep`` for images with `SparseResampleImage` in place of
    drizzle.  The parameters are jwst's, with these differences:

    - no ``kernel``: the square kernel only;
    - no ``propagate_dq``: the DQ array is not resampled;
    - no ``good_bits``: the pixels with DO_NOT_USE or NON_SCIENCE are left
      out (`RESAMPLE_REJECTED`), as by jwst's step;
    - ``weight_type`` is "ivm" unless given: the step does not read jwst's
      step parameters from CRDS, which ask for "exptime" for some
      instruments;
    - ``enable_ctx`` is off by default;
    - ``pixmap_stepsize`` 16 and ``pixmap_order`` 3: the pixel map is
      interpolated with splines between the pixels where the WCS is
      evaluated;
    - ``variance``: "exact" for the exact variance planes and the noise
      factor, "stcal" for the jwst step's variance planes and ERR.

    With the exact variance, the noise factor is saved as an image next to
    the mosaic, with the suffix ``nfactor``, whenever the results are saved,
    and its file name is recorded in the mosaic as
    ``meta.snowblind.sparse_resample.noise_factor_file`` (``NFACFILE``).
    """

    class_alias = "sparse_resample"
    fits_keyword = "S_SPARSE"
    fits_comment = "Resampling: snowblind.SparseResampleStep"

    spec = """
        pixfrac = float(min=0.0, max=1.0, default=1.0)  # Pixel shrinkage factor
        fillval = string(default='NAN')  # Output value for pixels with no weight, a number
        weight_type = option('ivm', 'exptime', default='ivm')  # Input image weighting type
        output_shape = int_list(min=2, max=2, default=None)  # [x, y] order
        crpix = float_list(min=2, max=2, default=None)  # 0-based image coordinates of the reference pixel
        crval = float_list(min=2, max=2, default=None)  # world coordinates of the reference pixel
        rotation = float(default=None)  # Output image Y-axis PA relative to North
        pixel_scale_ratio = float(default=1.0)  # Ratio of output to input pixel scale.
        pixel_scale = float(default=None)  # Absolute pixel scale in arcsec
        output_wcs = string(default='')  # Custom output WCS. Overrides other WCS parameters if provided.
        single = boolean(default=False)  # Resample each input to its own output grid
        blendheaders = boolean(default=True)  # Blend metadata from inputs into output
        in_memory = boolean(default=True)  # Keep images in memory
        enable_ctx = boolean(default=False)  # Compute and report the context array
        enable_err = boolean(default=True)  # Compute and report the err array
        report_var = boolean(default=True)  # Report the variance array
        variance = option('exact', 'stcal', default='exact')  # Exact variance and noise factor, or stcal's variance
        pixmap_stepsize = integer(min=1, default=16)  # Spacing in pixels of the WCS evaluations for the pixel map
        pixmap_order = integer(default=3)  # Spline order of the pixel map between them, 1 or 3
    """  # noqa: E501

    reference_file_types: list = []

    def process(self, input_data):
        """Resample the input images, as jwst's ResampleStep does.

        Parameters
        ----------
        input_data : str, `~stdatamodels.jwst.datamodels.ImageModel`, or any \
                     association-type input loadable into `~jwst.datamodels.library.ModelLibrary`
            An image file or model, or an association of them.

        Returns
        -------
        `~stdatamodels.jwst.datamodels.ImageModel` or `~jwst.datamodels.library.ModelLibrary`
            The mosaic, or with ``single`` a library of one resampled image per group.
        """
        input_model = self.prepare_output(input_data, open_models=False)
        if isinstance(input_model, ModelLibrary):
            input_models = input_model
        elif isinstance(input_model, ImageModel) or (
            isinstance(input_model, str) and filetype.check(input_model) in ["fits", "asdf"]
        ):
            input_models = ModelLibrary([input_model], on_disk=not self.in_memory)
            self.blendheaders = False
        elif isinstance(input_model, (str, dict, list, ModelContainer)):
            input_models = ModelLibrary(input_model, on_disk=not self.in_memory)
        else:
            raise TypeError(f"Input {input_data} is not a 2D image.")

        try:
            output = input_models.asn["products"][0]["name"]
        except KeyError:
            output = None

        with input_models:
            for model in input_models:
                if model.data.ndim != 2:
                    raise RuntimeError(f"Input {model.meta.filename} is not a 2D image.")
                # make NaN and DO_NOT_USE agree in all inputs, as the jwst step does
                match_nans_and_flags(model)
                input_models.shelve(model)

        kwargs = self._resample_kwargs()
        if self.single:
            resamp = SparseResampleImage(
                input_models, output=output, enable_ctx=self.enable_ctx, enable_var=False, compute_err="driz_err",
                **kwargs,
            )
            return resamp.resample_many_to_many(in_memory=self.in_memory)

        resamp = SparseResampleImage(
            input_models,
            output=output,
            variance=self.variance,
            enable_ctx=self.enable_ctx,
            enable_var=self.enable_err,
            report_var=self.enable_err and self.report_var,
            compute_err="from_var" if self.enable_err else None,
            **kwargs,
        )
        result = resamp.resample_many_to_one()
        if self.enable_err:
            self.record(result, variance=self.variance)
        self._save_noise_factor(result, resamp.noise_factor)
        return result

    def _mark_complete(self, model):
        """Record this step's status in an output model."""
        self.set_cal_step(model, "COMPLETE")

    def _resample_kwargs(self):
        """The SparseResampleImage parameters from the step parameters, as jwst's ResampleStep.get_drizpars."""
        output_shape = ResampleStep.check_list_pars(self.output_shape, "output_shape", min_vals=[1, 1])
        return {
            "pixfrac": self.pixfrac,
            "fillval": self.fillval,
            "weight_type": self.weight_type,
            "rejected": RESAMPLE_REJECTED,
            "on_output": self._mark_complete,
            "blendheaders": self.blendheaders,
            "pixmap_stepsize": self.pixmap_stepsize,
            "pixmap_order": self.pixmap_order,
            "output_wcs": load_custom_wcs(self.output_wcs, output_shape),
            "wcs_pars": {
                "crpix": ResampleStep.check_list_pars(self.crpix, "crpix"),
                "crval": ResampleStep.check_list_pars(self.crval, "crval"),
                "rotation": self.rotation,
                "pixel_scale": self.pixel_scale,
                "pixel_scale_ratio": self.pixel_scale_ratio,
                "output_shape": None if output_shape is None else output_shape[::-1],
            },
        }

    def _save_noise_factor(self, result, noise_factor):
        """Save the noise factor as an image next to the mosaic, if results are saved, and record its file name.

        Parameters
        ----------
        result : `~stdatamodels.jwst.datamodels.ImageModel`
            The mosaic, whose primary header and WCS the noise factor image gets.
        noise_factor : ndarray or None
            The noise factor; None saves nothing.
        """
        if noise_factor is None:
            return
        with ImageModel(noise_factor) as model:
            model.update(result, only="PRIMARY")
            model.meta.wcs = result.meta.wcs
            model.meta.wcsinfo = result.meta.wcsinfo
            path = self.save_model(model, suffix=NOISE_FACTOR_SUFFIX)
        if path is not None:
            log.info(f"Saved the noise factor in {path}")
            self.record(result, noise_factor_file=Path(path).name)
