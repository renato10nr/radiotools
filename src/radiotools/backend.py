# -*- coding: utf-8 -*-
"""backend module definitions.

This module creates the classed and method needed for the definition of a instrument as a **radiotelescope**.

Example:
    Two instruments are predefined as instances of this class. **Uirapuru** and **Callisto**.

"""
import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import scipy.signal
from astropy.io import fits
from astropy.constants import c
from astropy.constants import k_B
from abc import ABC, abstractmethod

TCOLD =  273.15 + 30


class Backend(ABC):
    """Backends for radiotelescope are abstract. Explicit instantes are callistospectrometer, usrp, rtlsdr."""

    def __init__(self,
                name=None,
                path=None,
                filetype = None,
                modes = None,
                nominal_slope = None,
                bandwidth = None,
                integration_time = None,
                temperature = None,
                gain = None,
                DCOLD = None,
                DWARM = None,
                DHOT = None,
                ENR_hot = None,
                ENR_warm = None):
        """Instantiate and go."""
        self._name = name
        self._path = path
        self.modes = modes
        self._nominal_slope = nominal_slope
        self._bandwidth = bandwidth
        self._integration_time = integration_time
        self._temperature = temperature
        self._gain = gain
        self._DCOLD = DCOLD
        self._WARM = DWARM
        self._DHOT = DHOT
        self._ENR_hot = ENR_hot
        self._ENR_warm = ENR_warm
        self._filetype = filetype
        self.filenames = None
        self.slope = None
        self.NF = None
        self.last_cal = None
        self.time_cold = None
        self.freqs = None
        self.Dhot = None
        self.Dwarm = None
        self.Dcold = None


    @property
    def name(self):
        """Return the name of the backend."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name of the backend."""
        self._name = name

    @property
    def path(self):
        """Return the path of the backend."""
        return self._path

    @path.setter
    def path(self, path):
        """Set the path of the backend."""
        self._path = path

    @property
    def filetype(self):
        """Return the filetype of the backend."""
        return self._filetype

    @filetype.setter
    def filetype(self, filetype):
        """Set the filetype of the backend."""
        self._filetype = filetype

    @property
    def nominal_slope(self):
        """Return the nominal_slope of the backend."""
        return self._nominal_slope

    @nominal_slope.setter
    def nominal_slope(self, nominal_slope):
        """Set the nominal_slope of the backend."""
        self._nominal_slope = nominal_slope

    @property
    def bandwidth(self):
        """Return the bandwidth of the backend."""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        """Set the bandwidth of the backend."""
        self._bandwidth = bandwidth

    @property
    def integration_time(self):
        """Return the integration_time of the backend."""
        return self._integration_time

    @integration_time.setter
    def integration_time(self, integration_time):
        """Set the integration_time of the backend."""
        self._integration_time = integration_time

    @property
    def temperature(self):
        """Return the temperature of the backend."""
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        """Set the temperature of the backend."""
        self._temperature = temperature

    @property
    def gain(self):
        """Return the gain of the backend."""
        return self._gain

    @gain.setter
    def gain(self, gain):
        """Set the gain of the backend."""
        self._gain = gain

    @property
    def _DHOT(self):
        """Return the _DHOT of the backend."""
        return self.__DHOT

    @_DHOT.setter
    def _DHOT(self, _DHOT):
        """Set the _DHOT of the backend."""
        self.__DHOT = _DHOT

    @property
    def _DWARM(self):
        """Return the _DWARM of the backend."""
        return self.__DWARM

    @_DWARM.setter
    def _DWARM(self, _DWARM):
        """Set the _DWARM of the backend."""
        self.__DWARM = _DWARM

    @property
    def _DCOLD(self):
        """Return the _DCOLD of the backend."""
        return self.__DCOLD

    @_DCOLD.setter
    def _DCOLD(self, _DCOLD):
        """Set the _DCOLD of the backend."""
        self.__DCOLD = _DCOLD

    @property
    def ENR_hot(self):
        """Return the ENR_hot of the backend."""
        return self._ENR_hot

    @ENR_hot.setter
    def ENR_hot(self, ENR_hot):
        """Set the ENR_hot of the backend."""
        self._ENR_hot = ENR_hot

    @property
    def ENR_warm(self):
        """Return the ENR_warm of the backend."""
        return self._ENR_warm

    @ENR_warm.setter
    def ENR_warm(self, ENR_warm):
        """Set the ENR_warm of the backend."""
        self._ENR_warm = ENR_warm

    @abstractmethod
    def _get_files_timestamps(self):
        pass

    @abstractmethod
    def load_data(self, filenames = None):
        pass

    @abstractmethod
    def _from_digits_to_mV(self, df=None):
        pass

    @abstractmethod
    def _calibrate_slope(self):
        pass

    @abstractmethod
    def calibrate(self, data=None, dcold = None):
        pass


class CallistoSpectrometer(Backend):
    """Short summary."""

    def __init__(self,
                name=None,
                path=None,
                filetype = "fit",
                modes = {"COLD":"04", "WARM":"02", "HOT":"03", "SKY":"01"},
                nominal_slope = 25.4,
                bandwidth = 300000,
                integration_time = 0.001,
                temperature = TCOLD,
                gain = 1,
                DCOLD = 170,
                DWARM = 185,
                DHOT = 210,
                ENR_hot = 15.0,
                ENR_warm = 5.0):
        """Instantiate and go."""
        self._name = name
        self._path = path
        self.modes = modes
        self._nominal_slope = nominal_slope
        self._bandwidth = bandwidth
        self._integration_time = integration_time
        self._temperature = temperature
        self._gain = gain
        self._DCOLD = DCOLD
        self._WARM = DWARM
        self._DHOT = DHOT
        self._ENR_hot = ENR_hot
        self._ENR_warm = ENR_warm
        self._filetype = filetype
        self.filenames = None
        self.slope = None
        self.NF = None
        self.last_cal = None
        self.time_cold = None
        self.freqs = None
        self.Dhot = None
        self.Dwarm = None
        self.Dcold = None

    def _get_files_timestamps(self):
        """Read all the files in PATH,"""
        path = self.path
        modes = self.modes
        instrument = self.name
        filetype = self.filetype
        # strip info from filenames and store in dataframe.
        for mode, value in modes.items():
            filenames = glob(path + instrument + "*_" + value + "." + filetype)
            df = pd.DataFrame({'files': filenames})
            df["mode"] = mode
        # strip timestamps from filenames
        df['timestamps'] = df.files.apply(lambda row: "T".join(row.split('/')[-1].split('_')[1:3]))
        # create time index from pandas datetime in utc scale.
        df['timestamps'] = pd.to_datetime(df['timestamps'], format = "%Y%m%dT%H%M%S", utc=True)
        self.filenames = df.set_index('timestamps').sort_index()
        return self


    def load_data(self, filenames = None):
        """Load the list of `filenames` and returns the approriate dataframe."""
        timestamp = []
        hdu_data = []
        timevector = []
        for file in filenames:
            with fits.open(file) as hdul:
                stamp = pd.to_datetime(hdul[0].header['DATE-OBS'] + "T" + hdul[0].header['TIME-OBS'])
                timestamp.append(stamp)
                data = hdul[0].data
                hdu_data.append(data)
                freqs = hdul[1].data[0][1]
                times = hdul[1].data[0][0]
                vector = stamp + pd.to_timedelta(times, unit="s")
                timevector.append(vector)
        # Union of DatetimeIndexes
        times = pd.DatetimeIndex(np.unique(np.hstack(timevector)))
        # Stack digit data and frequency goes in columns
        data = np.hstack(hdu_data).T
        df = pd.DataFrame(data, columns=freqs, index = times)
        # Discard 10 lowest frequency channels
        df = df.drop(df.iloc[:, [-10, -1]], axis=1)
        # Reorder freqs
        df = df[sorted(df.columns.tolist())]
        # That is important to consider properly the missing data.
        # Set periodicity as Callisto
        df = df.asfreq(freq='0.5S')
        return df


    def _from_digits_to_mV(self, df=None):
        """Pretty much what it does."""
        df = df*2500./255.  #floating point to ensure proper broadcasting
        return df

    def _calibrate_slope(self):
        """Slope in mV/dB for callisto from calibrated noise sources."""
        try:
            last_hot = self.filenames.query("mode == 'HOT'").iloc[[-1]]
            last_warm = self.filenames.query("mode == 'WARM'").iloc[[-1]]
            time_hot = last_hot.index[0]
            time_warm = last_warm.index[0]
            if abs(time_hot - time_warm) < pd.Timedelta(hours = 12):
                hot_data = self._load_data(filenames = last_hot.iloc[[0]]["files"].tolist())
                warm_data = self._load_data(filenames = last_warm.iloc[[0]]["files"].tolist())
                # factor 10 comes from dB scale.
                self.freqs = np.array(hot_data.median().index)
                self.Dhot = np.array(hot_data.median())
                self.Dwarm = np.array(warm_data.median())
                slope = _from_digits_to_mV(self.Dhot - self.Dwarm)/10.0
                size = slope.shape[0]
                windows = 10
                slope = savgol_filter(slope, 2 * np.floor(size/2/windows) + 1, 2, mode = "nearest")
                last_cal = abs(pd.to_datetime("today") - time_hot).days
                self.last_cal = last_cal
                print("Callibrating slope: {:2f} mV/dB \nCalibration is {} days old".format(slope, last_cal))
            else:
                print("HOT and WARM measurements are more than 12h apart. No callibration is done. Nominal value is set.")
                slope = self.nominal_slope
        except IndexError as e:
            print("Could not find suitable files to calibrate. Nominal value is set.")
            slope = self.nominal_slope
            self.Dhot = self._DHOT
            self.Dwarm = self._WARM
        self.slope = slope
        return self

    def calibrate(self, data=None, dcold = None):
        """Calibrate data em dBm"""
        ##
        freqs = np.array(data.columns)
        size = freqs.size
        ENR_warm = self.ENR_warm
        ENR_hot = self.ENR_hot
        YcdB = self._from_digits_to_mV(self.Dhot-self.Dwarm)/self.slope
        Yc = pow(10, YcdB/10)
        NF = ENR_warm - 10 * np.log10(Yc-1)
        # Set DCOLD
        if dcold is None:
            try:
                last_cold = self.filenames.query("mode == 'COLD'").iloc[[-1]]
                self.time_cold = last_cold.index[0]
                self.Dcold = last_cold.median()
                last_cal = abs(pd.to_datetime("today") - self.time_cold).days
                print("Last cold calibration is {} days old".format(last_cal))
            except IndexError as e:
                print("Could not find suitable file to calibrate. Using constante background as 170 digits. Provide Dcold dataframe as argument of the function manually instead.")
                self.Dcold = self._DCOLD * np.ones(size)
        else:
            last_cold = dcold
            self.time_cold = last_cold.index[0]
            self.Dcold = last_cold.median()
        #----
        Trx = self.temperature * ( pow(10, NF/10.) - 1.)

        Ys = pow(10, self._from_digits_to_mV(data - self.Dcold) / self.slope / 10)
        Trfi = Trx * (Ys - 1) + Ys * self.temperature
        Aeff = self.gain * pow( c/(freqs * 1000000), 2) /(4. * np.pi)
        # power in mW
        S_flux = 1000 * (2. * k_B / np.sqrt(self.bandwidth * self.integration_time)) * Trfi/Aeff
        # power in dBm
        SdB = 10.0 * np.log10(S_flux)
        # Union of DatetimeIndexes
        times = data.index
        df = pd.DataFrame(SdB, columns=freqs, index = times)
        return df
