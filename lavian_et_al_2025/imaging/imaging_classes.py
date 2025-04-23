from bouter import MultiSessionExperiment, EmbeddedExperiment
import json
import numpy as np
from split_dataset import SplitDataset


class ImagingExperiment:
    """
    Abstract base class for imaging experiment data.

    This class provides a common interface for different types of imaging experiments
    (two-photon microscopy, light-sheet microscopy), handling metadata and
    providing access to common imaging parameters.

    Parameters
    ----------
    dt_imaging : float, optional
        Time step between frames in seconds
    resolution : tuple, optional
        Spatial resolution (z, x, y) in microns per pixel
    indicator : str, optional
        Calcium indicator type (e.g., 'GCaMP6f', 'GCaMP6s')
    n_planes : int, optional
        Number of imaging planes
    *args, **kwargs :
        Additional arguments passed to parent class
    """
    def __init__(self, *args, dt_imaging=None, resolution=None,
                 indicator=None, n_planes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scope_config = self["imaging"]["microscope_config"]

        self._dt_imaging = dt_imaging
        self._resolution = resolution
        self._indicator = indicator
        self._n_planes = n_planes
        self._data_shape = None
        self._fn = None

    @property
    def data_shape(self):
        if self._data_shape is None:
            self._data_shape = SplitDataset(s)
        return self._data_shape

    @property
    def dt_imaging(self):
        pass

    @property
    def fn(self):
        pass

    @property
    def resolution(self):
        pass

    @property
    def n_planes(self):
        pass

    @property
    def fs_imaging(self):
        return 1 / self.dt_imaging


class TwoPExperiment(ImagingExperiment, MultiSessionExperiment):
    """
    Class for handling two-photon imaging experiment data collected using Brunoise (software).

    Extends ImagingExperiment with specific functionality for two-photon microscopy,
    including metadata parsing for resolution calculation and frame timing.

    Parameters
    ----------
    path : Path
        Path to the experiment directory
    stack_shape : tuple, optional
        Shape of the image stack (t, z, x, y)
    *args, **kwargs :
        Additional arguments passed to parent classes
    """
    def __init__(
        self, path, stack_shape=None, *args, **kwargs
    ):
        path = path / "behavior"
        super().__init__(path, *args, **kwargs)
        self.imaging_type = "2p"
        self.path = path.parent
        self.stack_shape = stack_shape

    @property
    def dt_imaging(self):
        if self._dt_imaging is None:
            try: # for data acquired with old LabView software
                self._dt_imaging = self.scope_config["frame_time"] / 1000
            except KeyError:  # for data acquired with Brunoise
                self._dt_imaging = 1 / self.scope_config["scanning"]["framerate"]

        return self._dt_imaging

    @property
    def n_planes(self):
        return len(self.session_list)

    @property
    def resolution(self):
        if self._resolution is None:
            if self.stack_shape is None:
                metadata_file_stack = self.path / "original/stack_metadata.json"
                with open(str(metadata_file_stack), "r") as f:
                    stack_param = json.load(f)
                    n_px_x, n_px_y = stack_param["shape_full"][2:4]
            else:
                _, n_px_x, n_px_y = self.stack_shape

            z_res = self.scope_config["recording"]["dz"]
            aspect_ratio = self.scope_config["scanning"]["aspect_ratio"]
            voltage_max = self.scope_config["scanning"]["voltage"]
            if aspect_ratio >= 1:
                voltage_x = voltage_max
                voltage_y = voltage_x / aspect_ratio
            else:
                voltage_x = voltage_max * aspect_ratio
                voltage_y = voltage_max

            x_res = compute_resolution(voltage_x, n_px_x)
            y_res = compute_resolution(voltage_y, n_px_y)
            self._resolution = (z_res, x_res, y_res)

        return self._resolution


class LightsheetExperiment(ImagingExperiment, EmbeddedExperiment):
    """
    Class for handling light-sheet imaging experiment data collected using Sashimi (software).

    Extends ImagingExperiment with specific functionality for light-sheet microscopy,
    including metadata parsing for z-scanning frequency and resolution calculation.

    Parameters
    ----------
    path : Path
        Path to the experiment directory
    *args, **kwargs :
        Additional arguments passed to parent classes
    """
    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self.imaging_type = "lightsheet"

    @property
    def fn(self):
        if self._fn is None:
            try:
                self._fn = self.scope_config["piezo_z"]["frequency"]
            except KeyError:
                pass

            try:
                self._fn = self.scope_config["lightsheet"]["scanning"]["z"]["frequency"]
            except KeyError:
                pass

        if self._fn is None:
            raise ValueError("No entries for frequency have been found in metadata!")

        return self._fn

    @property
    def dt_imaging(self):
        return 1 / self.fn

    @property
    def n_planes(self):
        if self._n_planes is None:
            try:
                self._n_planes = self.scope_config["camera_trigger"]["n_planes"]
            except KeyError:
                try:
                    self._n_planes = len(
                        self.scope_config["camera_trigger"]["pulse_times"])
                except KeyError:
                    self._n_planes = self.scope_config["lightsheet"]["triggering"]["n_planes"]

        return self._n_planes

    @property
    def resolution(self):
        if self._resolution is None:
            try:
                z_tot_span = self.scope_config["piezo_z"]["amplitude"] * 2
            except KeyError:
                try:
                    lsconfig = self.scope_config["lightsheet"]
                    z_tot_span = lsconfig["z"]["piezo_max"] - lsconfig["z"]["piezo_min"]
                except KeyError:
                    raise Warning("No valid entry for z span found in metadata")

            z_span = z_tot_span / self.n_planes
            self._resolution = (z_span, 0.6, 0.6)

        return self._resolution


def compute_resolution(zoom, size_px):
    """
    Calculates pixel resolution for two-photon imaging data.

    Uses calibration data to convert between zoom factor and physical distance.

    Parameters
    ----------
    zoom : float
        Zoom parameter from microscope software
    size_px : int
        Image size in pixels on the matching axis

    Returns
    -------
    float
        Pixel size in microns

    Note:
    The calculation uses a calibration based on known distances measured
    at different zoom levels, scaling proportionally to the current zoom
    and image size.
    """
    # Calibration data:
    dist_in_um = 10
    dist_in_px = np.array([21.13, 19.62, 8.93])
    zooms = np.array([1.5, 3, 4.5])
    image_max_sizes = np.array([330, 610, 410])

    return np.mean(
        (dist_in_um / dist_in_px) * (zoom / zooms) * (image_max_sizes / size_px)
    )