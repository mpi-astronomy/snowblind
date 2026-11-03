"""The Image3 pipeline with snowblind's outlier detection and spizzle's resampling in place of the pipeline's own.

``calwebb_image3`` drizzles twice: to find outliers and to make the
mosaic.  This pipeline keeps every other step of the jwst pipeline and
puts `~snowblind.outlier_reference.TiledOutlierStep` (or, with
``tiled_outliers`` off, `~snowblind.outlier_detection.SparseOutlierDetectionStep`)
and `~snowblind.resample.SparseResampleStep` in those two slots.  It runs as
``strun snowblind_image3`` or through `Image3Pipeline.call`, and needs
``pip install 'snowblind[spizzle]'``.
"""
from jwst.pipeline import Image3Pipeline as JwstImage3Pipeline

from .outlier_detection import SparseOutlierDetectionStep
from .outlier_reference import TiledOutlierStep
from .pipeline import _as_bool, _parameters_unknown_to
from .resample import SparseResampleStep

__all__ = ["Image3Pipeline"]


class Image3Pipeline(JwstImage3Pipeline):
    """``calwebb_image3`` with snowblind's outlier detection and resampling by spizzle.

    The step slots are the jwst pipeline's, so the replaced steps take
    their parameters as ``--steps.outlier_detection.scale=...`` and
    ``--steps.resample.pixel_scale=...``.  They take none from CRDS, and
    accept only their own parameters (no ``kernel``, for one).  The other
    steps, and the parameters CRDS provides for them, are jwst's.

    The ``outlier_detection`` slot holds `TiledOutlierStep`, which compares
    each exposure with one reference built from all the exposures' input
    pixels; ``--tiled_outliers=False`` puts `SparseOutlierDetectionStep`
    there instead, jwst's method with spizzle's drizzling.  With the exact
    variance (the default), the mosaic's noise factor is saved next to it as
    ``<product>_nfactor.fits``.
    """

    class_alias = "snowblind_image3"

    spec = JwstImage3Pipeline.spec + """
    tiled_outliers = boolean(default=True)  # find outliers with TiledOutlierStep; False uses SparseOutlierDetectionStep
    """

    step_defs = {
        **JwstImage3Pipeline.step_defs,
        "outlier_detection": TiledOutlierStep,
        "resample": SparseResampleStep,
    }

    def __init__(self, *args, **kwargs):
        """Create the pipeline, with `SparseOutlierDetectionStep` in the outlier slot when ``tiled_outliers`` is off.

        Parameters
        ----------
        *args
            Passed to the jwst pipeline.
        **kwargs
            Passed to the jwst pipeline.  With ``tiled_outliers`` off (a
            boolean, or the string of one that strun or a config file
            gives), the ``outlier_detection`` slot holds
            `SparseOutlierDetectionStep`, the parameters it does not know
            are dropped from the ``outlier_detection`` entry of ``steps``,
            and those whose defaults differ between the two steps (``scale``)
            take the sparse step's default unless that entry gives them.
        """
        sparse = not _as_bool(kwargs.get("tiled_outliers", True))
        given = set()
        if sparse:
            self.step_defs = {**self.step_defs, "outlier_detection": SparseOutlierDetectionStep}
            # Step.call builds the outlier_detection section from this class's spec, which is TiledOutlierStep's; the
            # sparse step would reject the parameters it does not know, so they are dropped from the section here
            section = (kwargs.get("steps") or {}).get("outlier_detection")
            if section is not None:
                for key in _parameters_unknown_to(SparseOutlierDetectionStep, section):
                    section.pop(key, None)
                # a ConfigObj section lists the values that came from the spec's defaults
                given = set(section) - set(getattr(section, "defaults", ()))
        super().__init__(*args, **kwargs)
        if sparse:
            tiled_defaults, sparse_defaults = TiledOutlierStep(), SparseOutlierDetectionStep()
            for key in SparseOutlierDetectionStep.load_spec_file():
                default = getattr(sparse_defaults, key, None)
                if key not in given and hasattr(tiled_defaults, key) and getattr(tiled_defaults, key) != default:
                    setattr(self.outlier_detection, key, default)
