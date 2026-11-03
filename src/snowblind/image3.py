"""The Image3 pipeline with spizzle's outlier detection and resampling in place of the pipeline's own.

``calwebb_image3`` drizzles twice: to find outliers and to make the
mosaic.  This pipeline keeps every other step of the jwst pipeline and
puts `~snowblind.outlier_detection.SparseOutlierDetectionStep` and
`~snowblind.resample.SparseResampleStep` in those two slots.  It runs as
``strun snowblind_image3`` or through `Image3Pipeline.call`, and needs
``pip install 'snowblind[spizzle]'``.
"""
from jwst.pipeline import Image3Pipeline as JwstImage3Pipeline

from .outlier_detection import SparseOutlierDetectionStep
from .resample import SparseResampleStep

__all__ = ["Image3Pipeline"]


class Image3Pipeline(JwstImage3Pipeline):
    """``calwebb_image3`` with outlier detection and resampling by spizzle.

    The step slots are the jwst pipeline's, so the two replaced steps take
    their parameters as ``--steps.outlier_detection.pixfrac=...`` and
    ``--steps.resample.pixel_scale=...``.  They take none from CRDS, and
    accept only their own parameters (no ``kernel``, for one).  The other
    steps, and the parameters CRDS provides for them, are jwst's.  With the
    exact variance (the default), the mosaic's noise factor is saved next
    to it as ``<product>_nfactor.fits``.
    """

    class_alias = "snowblind_image3"

    step_defs = {
        **JwstImage3Pipeline.step_defs,
        "outlier_detection": SparseOutlierDetectionStep,
        "resample": SparseResampleStep,
    }
