"""Resampling of images with spizzle in place of drizzle.

`SparseResampleImage` is jwst's `~jwst.resample.resample.ResampleImage`
with drizzle's square kernel replaced by spizzle's sparse operator: the
pixel map, the operator of each exposure, and the sums of the science data,
the weights and the variance planes, in float64.  The weights are stcal's
(``build_driz_weight``), and everything else, from the output grid to the
metadata, is jwst's and stcal's.

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
which call ``super()``, reach it.  It uses private attributes of stcal's
class and was checked against stcal 1.20.0 and jwst 3.0.0.

spizzle is an optional dependency, ``pip install 'snowblind[spizzle]'``;
without it this module imports, and `SparseResampleImage` raises.
"""
import logging

import numpy as np
from jwst.resample.resample import ResampleImage, copy_asn_info_from_library
from stcal.resample import Resample
from stcal.resample.utils import build_driz_weight, get_tmeasure, is_imaging_wcs, resample_range

try:
    import spizzle
except ImportError:
    spizzle = None

__all__ = ["SparseResampleImage", "VARIANCE_MODES"]

log = logging.getLogger(__name__)

# the stcal release whose Resample internals _SpizzleResample was checked against
STCAL_CHECKED_VERSION = "1.20.0"

VARIANCE_MODES = ("exact", "stcal")


class _SpizzleResample(Resample):
    """stcal's Resample with spizzle's sums in place of drizzle's.

    ``variance`` must be set before ``Resample.__init__`` runs, as
    `SparseResampleImage` does.
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
                "before finalize, as resample_many_to_one does."
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
        """spizzle's operator of one exposure onto the output grid, with stcal's weights and input range."""
        shape = model["data"].shape
        pixmap = spizzle.pixel_map(
            model["wcs"], self.output_wcs, shape=shape, stepsize=max(1, int(self.pixmap_stepsize)), order=self.pixmap_order
        )
        weights = build_driz_weight(
            model, weight_type=self.weight_type, good_bits=self.good_bits, flag_name_map=self.dq_flag_name_map
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
    **kwargs
        The other parameters of jwst's ``ResampleImage``.  ``kernel`` must be
        "square", and ``propagate_dq`` is not supported.  ``fillval`` must be
        a number or a string of one, such as "NAN".
    """

    def __init__(
        self, input_models, variance="exact", pixmap_stepsize=16, pixmap_order=3, enable_ctx=False, **kwargs
    ):
        if spizzle is None:
            raise ImportError("SparseResampleImage needs spizzle: pip install 'snowblind[spizzle]'")
        if variance not in VARIANCE_MODES:
            raise ValueError(f"variance must be one of {VARIANCE_MODES}, not {variance!r}")
        if (kernel := kwargs.get("kernel", "square")) != "square":
            raise ValueError(f"SparseResampleImage has only the square kernel, not {kernel!r}")
        if kwargs.get("propagate_dq", False):
            raise ValueError("SparseResampleImage does not propagate DQ")
        self.variance = variance
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

    def resample_many_to_one(self):
        """Resample and co-add all the input models to one output, with a second pass for the noise factor.

        Returns
        -------
        `~stdatamodels.jwst.datamodels.ImageModel`
            The resampled and co-added image.
        """
        log.info("Resampling science and variance data with spizzle")
        with self.input_models:
            for model in self.input_models:
                self.add_model(model)
                self.input_models.shelve(model, modify=False)
            if self._exact_variance():
                log.info("Computing the noise factor in a second pass over the exposures")
                for model in self.input_models:
                    self.add_noise_factor(model)
                    self.input_models.shelve(model, modify=False)

        self.finalize()
        self.output_jwst_model.meta.filename = self.output_filename
        copy_asn_info_from_library(self.input_models, self.output_jwst_model)
        return self.output_jwst_model
