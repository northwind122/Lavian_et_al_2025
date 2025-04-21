from bouter import MultiSessionExperiment, EmbeddedExperiment
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
import warnings
from split_dataset import SplitDataset


class ImagingExperiment:
    """ General class to deal with imaging experiments.
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
    def tau_off(self):
        return sensor_taus[self.indicator]

    @property
    def fs_imaging(self):
        return 1 / self.dt_imaging

class TwoPExperiment(ImagingExperiment, MultiSessionExperiment):
    """ Class to deal with 2P metadata reading.
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
            try:
                self._dt_imaging = self.scope_config["frame_time"] / 1000
            except KeyError:  # for data acquired with the python 2P program
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
            try:  # new python software data
                z_res = self.scope_config["recording"]["dz"]
                aspect_ratio = self.scope_config["scanning"]["aspect_ratio"]
                voltage_max = self.scope_config["scanning"]["voltage"]
                if aspect_ratio >= 1:
                    voltage_x = voltage_max
                    voltage_y = voltage_x / aspect_ratio
                else:
                    voltage_x = voltage_max * aspect_ratio
                    voltage_y = voltage_max

            except KeyError:  # old labview software data
                z_res = self.scope_config["z_step"]
                voltage_x = self.scope_config["z_indicator"]  #TODO check this
                voltage_y = self.scope_config["y_zoom"]

            x_res = compute_resolution(voltage_x, n_px_x)
            y_res = compute_resolution(voltage_y, n_px_y)
            self._resolution = (z_res, x_res, y_res)

        return self._resolution


class LightsheetExperiment(ImagingExperiment, EmbeddedExperiment):
    """ Class to deal with Lightsheet metadata reading.
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