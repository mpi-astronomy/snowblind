"""The Image3 pipeline with snowblind's outlier detection and spizzle's resampling in place of the pipeline's own.

``calwebb_image3`` drizzles twice: to find outliers and to make the
mosaic.  This pipeline keeps every other step of the jwst pipeline, finds
outliers with `~snowblind.outlier_reference.TiledOutlierStep` or
`~snowblind.outlier_detection.SparseOutlierDetectionStep`, and makes the
mosaic with `~snowblind.resample.SparseResampleStep`.  It runs as ``strun
snowblind_image3`` or through `Image3Pipeline.call`, and needs ``pip
install 'snowblind[spizzle]'``.
"""
from jwst.pipeline import Image3Pipeline as JwstImage3Pipeline

from .outlier_detection import SparseOutlierDetectionStep
from .outlier_reference import TiledOutlierStep
from .resample import SparseResampleStep

__all__ = ["Image3Pipeline"]


class Image3Pipeline(JwstImage3Pipeline):
    """``calwebb_image3`` with snowblind's outlier detection and resampling by spizzle.

    The jwst pipeline's ``outlier_detection`` slot is replaced by two:
    ``tiled_outlier``, `TiledOutlierStep`, which compares each exposure with
    one reference built from all the exposures' input pixels, and
    ``sparse_outlier``, `SparseOutlierDetectionStep`, jwst's method with
    spizzle's drizzling.  ``outlier_step`` names the one that runs
    (``tiled_outlier`` by default); each takes its own parameters, e.g.
    ``--steps.tiled_outlier.scale=...`` or ``--steps.sparse_outlier.pixfrac=...``.
    The ``resample`` slot holds `SparseResampleStep`
    (``--steps.resample.pixel_scale=...``), which saves the mosaic's noise
    factor next to it as ``<product>_nfactor.fits``.

    These steps take no parameters from CRDS and accept only their own (no
    ``kernel``, for one).  The other steps, and the parameters CRDS provides
    for them, are jwst's.
    """

    class_alias = "snowblind_image3"

    spec = JwstImage3Pipeline.spec + """
    outlier_step = option('tiled_outlier', 'sparse_outlier', default='tiled_outlier')  # the step that flags outliers
    """

    step_defs = {
        **{name: step for name, step in JwstImage3Pipeline.step_defs.items() if name != "outlier_detection"},
        "tiled_outlier": TiledOutlierStep,
        "sparse_outlier": SparseOutlierDetectionStep,
        "resample": SparseResampleStep,
    }

    def process(self, input_data):
        """Run the jwst pipeline, with the step that ``outlier_step`` names in its outlier detection.

        Parameters
        ----------
        input_data : Level3 Association, or `~jwst.datamodels.library.ModelLibrary`
            The exposures to process.
        """
        # the jwst pipeline sets up and runs the step it holds as outlier_detection
        self.outlier_detection = getattr(self, self.outlier_step)
        return super().process(input_data)
