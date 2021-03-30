# -*- coding: utf-8 -*-
"""
.. include:: ./README.md

"""

import copy
import pyvisa
import numpy as np
from typing import Tuple, List, Union


_VISA_ADDRESS = "USB0::0x0699::0x0353::1731975::INSTR"


def _SI_prefix_to_factor(unit):
    """Convert an SI prefix to a numerical factor

    Parameters
    ----------
    unit : str
        The unit whose first character is checked against the list of
        prefactors {"M": 1e6, "k": 1e3, "m": 1e-3}

    Returns
    -------
    factor : float or `None`
        The appropriate factor or 1 if not found in the list, or `None`
        if the unit string is empty
    """
    # SI prefix to numerical value
    SI_conversion = {"M": 1e6, "k": 1e3, "m": 1e-3}
    try:  # using the unit's first character as key in the dictionary
        factor = SI_conversion[unit[0]]
    except KeyError:  # if the entry does not exist
        factor = 1
    except IndexError:  # if the unit string is empty
        factor = None
    return factor


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ERROR CLASSES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


class NotSetError(Exception):
    """Error for when a value cannot be written to the instrument"""


class NotCompatibleError(Exception):
    """Error for when the instrument is not compatible with this module"""


## ~~~~~~~~~~~~~~~~~~~~~ FUNCTION GENERATOR CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


class FuncGen:
    """Class for interacting with Tektronix function generator

    Parameters
    ----------
    visa_address : str
        VISA address of the insrument
    impedance : tuple of {"highZ", "50ohm"}, default ("highZ",)*2
        Determines voltage limits associated with high impedance (whether the
        instrument is using 50ohm or high Z cannot be controlled through VISA).
        For example `("highZ", "50ohm")` for to use high Z for ch1 and
        50 ohm for ch2
    timeout : int, default 1000
        Timeout in milliseconds of instrument connection
    verify_param_set : bool, default False
        Verify that a value is successfully set after executing a set function
    verbose : bool, default `True`
        Choose whether to print information such as model upon connecting etc
    override_compatibility : str, default `""`
        If the instrument limits for the model connected to are not known
        `NotCompatibleError` will be raised. To override and use either of
        AFG1022, AFG1062, or AFG3022 limits, use their respecive model names as
        argument. Note that this might lead to unexpected behaviour for custom
        waveforms and 'MIN'/'MAX' keywords.

    Attributes
    ----------
    _visa_address : str
        The VISA address of the instrument
    _id : str
        Comma separated string with maker, model, serial and firmware of
        the instrument
    _inst : `pyvisa.resources.Resource`
        The PyVISA resource
    _arbitrary_waveform_length : list
        The permitted minimum and maximum length of an arbitrary waveform,
        e.g. [2, 8192]
    _arbitrary_waveform_resolution : int
        The vertical resolution of the arbitrary waveform, for instance 14 bit
        => 2**14-1 = 16383
    _max_waveform_memory_user_locations : int
        The number of the last user memory location available

    Raises
    ------
    pyvisa.Error
        If the supplied VISA address cannot be connected to
    NotCompatibleError
        If the instrument limits for the model connected to are not known
        (Call the class with `override_compatibility=True` to override and
        use AFG1022 limits)
    """

    _is_connected = False
    """bool: Keeping track of whether the PYVISA connection has been established"""

    instrument_limits = {}
    """dict: Contains the following keys with subdictionaries

        - `frequency lims`
          Containing the frequency limits for the instrument where the keys
          "min" and "max" have values corresponding to minimum and maximum
          frequencies in Hz
        - `voltage lims`
          Contains the maximum absolute voltage the instrument can output
          for the keys "50ohm" and "highZ" according to the impedance setting
        - `amplitude lims`
          Contains the smallest and largest possible amplitudes where the
          keys "50ohm" and "highZ" will have subdictionaries with keys
          "min" and "max"
    """

    def __init__(
        self,
        visa_address: str,
        impedance: Tuple[str, str] = ("highZ",) * 2,
        timeout: int = 1000,
        verify_param_set: bool = False,
        override_compatibility: str = "",
        verbose: bool = True,
    ):
        self._override_compat = override_compatibility
        self._visa_address = visa_address
        self.verify_param_set = verify_param_set
        """bool:  Verify that a value is successfully set after executing a set function"""
        self.verbose = verbose
        """bool: Choose whether to print information such as model upon connecting etc"""
        self.open(visa_address, timeout)
        self._initialise_model_properties()
        self.channels = (
            self._spawn_channel(1, impedance[0]),
            self._spawn_channel(2, impedance[1]),
        )
        """tuple of `FuncGenChannel`: Objects to control the channels"""
        self.ch1 = self.channels[0]
        """`FuncGenChannel`: Short hand for `channels[0]` Object to control channel 1"""
        self.ch2 = self.channels[1]
        """`FuncGenChannel`: Short hand for `channels[1]` Object to control channel 2"""

    def __enter__(self, **kwargs):
        # The kwargs will be passed on to __init__
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def open(self, visa_address: str, timeout: int):
        try:
            rm = pyvisa.ResourceManager()
            self._inst = rm.open_resource(visa_address)
        except pyvisa.Error:
            print(f"\nVisaError: Could not connect to '{visa_address}'")
            raise
        self._is_connected = True
        self.timeout = timeout
        # Clear all the event registers and queues used in the instrument
        # status and event reporting system
        self.write("*CLS")
        # Get information about the connected device
        self._id = self.query("*IDN?")
        # Second query might be needed due to unknown reason
        if self._id == "":
            self._id = self.query("*IDN?")
        self._maker, self._model, self._serial = self._id.split(",")[:3]
        if self.verbose:
            print(
                f"Connected to {self._maker} model {self._model}, "
                f"serial {self._serial}"
            )

    def close(self):
        """Close the connection to the instrument"""
        if self._is_connected:
            self._inst.close()
            self._is_connected = False

    @property
    def timeout(self) -> int:
        """The timeout of the PYVISA connection in milliseconds"""
        return self._inst.timeout

    @timeout.setter
    def timeout(self, ms: int):
        self._inst.timeout = ms

    def _initialise_model_properties(self):
        """Initialises the limits of what the instrument can handle according
        to the instrument model

        Raises
        ------
        NotCompatibleError
            If the connected model is not necessarily compatible with this
            package, slimits are not known.
        """
        if np.any(["AFG1022" in a for a in [self._model, self._override_compat]]):
            self.instrument_limits = {
                "frequency lims": ({"min": 1e-6, "max": 25e6}, "Hz"),
                "voltage lims": (
                    {"50ohm": {"min": -5, "max": 5}, "highZ": {"min": -10, "max": 10}},
                    "V",
                ),
                "amplitude lims": (
                    {
                        "50ohm": {"min": 0.001, "max": 10},
                        "highZ": {"min": 0.002, "max": 20},
                    },
                    "Vpp",
                ),
            }
            self._arbitrary_waveform_length = [2, 8192]  # min length, max length
            self._arbitrary_waveform_resolution = 16383  # 14 bit
            self._max_waveform_memory_user_locations = 255
        elif np.any(["AFG1062" in a for a in [self._model, self._override_compat]]):
            self.instrument_limits = {
                "frequency lims": ({"min": 1e-6, "max": 60e6}, "Hz"),
                "voltage lims": (
                    {"50ohm": {"min": -5, "max": 5}, "highZ": {"min": -10, "max": 10}},
                    "V",
                ),
                "amplitude lims": (
                    {
                        "50ohm": {"min": 0.001, "max": 10},
                        "highZ": {"min": 0.002, "max": 20},
                    },
                    "Vpp",
                ),
            }
            self._arbitrary_waveform_length = [2, 1e6]  # min length, max length
            self._arbitrary_waveform_resolution = 16383  # 14 bit
            self._max_waveform_memory_user_locations = 31
        elif np.any(["AFG3022" in a for a in [self._model, self._override_compat]]):
            self.instrument_limits = {
                "frequency lims": ({"min": 1e-6, "max": 25e6}, "Hz"),
                "voltage lims": (
                    {"50ohm": {"min": -5, "max": 5}, "highZ": {"min": -10, "max": 10}},
                    "V",
                ),
                "amplitude lims": (
                    {
                        "50ohm": {"min": 0.01, "max": 10},
                        "highZ": {"min": 0.02, "max": 20},
                    },
                    "Vpp",
                ),
            }
            self._arbitrary_waveform_length = [2, 65536]  # min length, max length
            self._arbitrary_waveform_resolution = 16383  # 14 bit
            self._max_waveform_memory_user_locations = 4
        else:
            msg = (
                f"Model {self._model} might not be fully supported!\n"
                "  The module has been tested with AFG1022, AFG1062, and AFG3022.\n"
                "  To initiate and use the module as any of these, call the\n"
                "  class with for instance `override_compatibility='AFG1022'`\n"
                "  Note that this might lead to unexpected behaviour\n"
                "  for custom waveforms and 'MIN'/'MAX' keywords."
            )
            raise NotCompatibleError(msg)

    def write(self, command: str, custom_err_message: str = None) -> int:
        """Write a VISA command to the instrument

        Parameters
        ----------
        command : str
            The VISA command to be written to the instrument
        custom_err_message : str, default `None`
            When `None`, the RuntimeError message is "Writing/querying command
            {command} failed: pyvisa returned StatusCode ..".
            Otherwise, if a message is supplied "Could not {message}:
            pyvisa returned StatusCode .."

        Returns
        -------
        bytes : int
            Number of bytes tranferred

        Raises
        ------
        RuntimeError
            If status returned by PyVISA write command is not
            `pyvisa.constants.StatusCode.success`
        """
        num_bytes = self._inst.write(command)
        self._check_pyvisa_status(command, custom_err_message=custom_err_message)
        return num_bytes

    def query(self, command: str, custom_err_message: str = None) -> str:
        """Query the instrument

        Parameters
        ----------
        command : str
            The VISA query command
        custom_err_message : str, default `None`
            When `None`, the RuntimeError message is "Writing/querying command
            {command} failed: pyvisa returned StatusCode ..".
            Otherwise, if a message is supplied "Could not {message}:
            pyvisa returned StatusCode .."

        Returns
        -------
        str
            The instrument's response

        Raises
        ------
        RuntimeError
            If status returned by PyVISA write command is not
            `pyvisa.constants.StatusCode.success`
        """
        response = self._inst.query(command).strip()
        self._check_pyvisa_status(command, custom_err_message=custom_err_message)
        return response

    def _check_pyvisa_status(self, command: str, custom_err_message: str = None):
        """Check the last status code of PyVISA

        Parameters
        ----------
        command : str
            The VISA write/query command

        Returns
        -------
        status : pyvisa.constants.StatusCode
            Return value of the library call

        Raises
        ------
        RuntimeError
            If status returned by PyVISA write command is not
            `pyvisa.constants.StatusCode.success`
        """
        status = self._inst.last_status
        if not status == pyvisa.constants.StatusCode.success:
            if custom_err_message is not None:
                msg = (
                    f"Could not {custom_err_message}: pyvisa returned "
                    f"StatusCode {status} ({str(status)})"
                )
                raise RuntimeError(msg)
            msg = (
                f"Writing/querying command {command} failed: pyvisa returned "
                f"StatusCode {status} ({str(status)})"
            )
            raise RuntimeError(msg)
        return status

    def get_error(self) -> str:
        """Get the contents of the Error/Event queue on the device

        Returns
        -------
        str
            Error/event number, description of error/event
        """
        return self.query("SYSTEM:ERROR:NEXT?")

    def _spawn_channel(self, channel: int, impedance: str) -> "FuncGenChannel":
        """Wrapper function to create a `FuncGenChannel` object for
        a channel -- see the class docstring"""
        return FuncGenChannel(self, channel, impedance)

    def get_settings(self) -> List[dict]:
        """Get dictionaries of the current settings of the two channels

        Returns
        -------
        settings : list of dicts
            [ch1_dict, ch2_dict]: Settings currently in use as a dictionary
            with keys output, function, amplitude, offset, and frequency with
            corresponding values
        """
        return [ch.get_settings() for ch in self.channels]

    def print_settings(self):
        """Prints table of the current setting for both channels"""
        settings = self.get_settings()
        # Find the necessary padding for the table columns
        # by evaluating the maximum length of the entries
        key_padding = max([len(key) for key in settings[0].keys()])
        ch_paddings = [
            max([len(str(val[0])) for val in ch_settings.values()])
            for ch_settings in settings
        ]
        padding = [key_padding] + ch_paddings
        print(f"\nCurrent settings for {self._maker} {self._model} {self._serial}\n")
        row_format = "{:>{padd[0]}s} {:{padd[1]}s} {:{padd[2]}s} {}"
        table_header = row_format.format("Setting", "Ch1", "Ch2", "Unit", padd=padding)
        print(table_header)
        print("=" * len(table_header))
        for (ch1key, (ch1val, unit)), (_, (ch2val, _)) in zip(
            settings[0].items(), settings[1].items()
        ):
            print(
                row_format.format(ch1key, str(ch1val), str(ch2val), unit, padd=padding)
            )

    def set_settings(self, settings: List[dict]):
        """Set the settings of both channels with settings dictionaries

        (Each channel is turned off before applying the changes to avoid
        potentially harmful combinations)

        Parameteres
        -----------
        settings : list of dicts
            List of settings dictionaries as returned by `get_settings`, first
            entry for channel 1, second for channel 2. The dictionaries should
            have keys output, function, amplitude, offset, and frequency
        """
        for ch, s in zip(self.channels, settings):
            ch.set_settings(s)

    def syncronise_waveforms(self):
        """Syncronise waveforms of the two channels when using the same frequency

        Note: Does NOT enable the frequency lock that can be enabled on the
        user interface of the instrument)
        """
        self.write(":PHAS:INIT", custom_err_message="syncronise waveforms")

    def get_frequency_lock(self) -> bool:
        """Check if frequency lock is enabled

        Returns
        -------
        bool
            `True` if frequency lock enabled
        """
        # If one is locked so is the other, so just need to check one
        return int(self.query("SOURCE1:FREQuency:CONCurrent?")) == 1

    def set_frequency_lock(self, state: str, use_channel: int = 1):
        """Enable the frequency lock to make the two channels have the same
        frequency and phase of their signals, also after adjustments.

        See also `FuncGen.syncronise_waveforms` for one-time sync only.

        Parameters
        ----------
        state : {"ON", "OFF"}
            ON to enable, OFF to disable the lock
        use_channel : int, default 1
            Only relevant if turning the lock ON: The channel whose frequency
            shall be used as the common freqency
        """
        if self.verbose:
            if state.lower() == "off" and not self.get_frequency_lock():
                print(
                    f"(!) {self._model}: Tried to disable frequency lock, but "
                    f"frequency lock was not enabled"
                )
                return
            if state.lower() == "on" and self.get_frequency_lock():
                print(
                    f"(!) {self._model}: Tried to enable frequency lock, but "
                    f"frequency lock was already enabled"
                )
                return
        # (Sufficient to disable for only one of the channels)
        cmd = f"SOURCE{use_channel}:FREQuency:CONCurrent {state}"
        msg = f"turn frequency lock {state}"
        self.write(cmd, custom_err_message=msg)

    def software_trig(self):
        """NOT TESTED: sends a trigger signal to the device
        (for bursts or modulations)"""
        self.write("*TRG", custom_err_message="send trigger signal")

    ## ~~~~~~~~~~~~~~~~~~~~~ CUSTOM WAVEFORM FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~ ##

    def get_waveform_catalogue(self) -> List[str]:
        """Get list of the waveforms that are in use (not empty)

        Returns
        -------
        catalogue : list
            Strings with the names of the user functions that are not empty
        """
        catalogue = self.query("DATA:CATalog?").split(",")
        catalogue = [wf[1:-1] for wf in catalogue]  # strip off extra quotes
        return catalogue

    def get_custom_waveform(self, memory_num: int) -> np.ndarray:
        """Get the waveform currently stored in USER<memory_num>

        Parameters
        ----------
        memory_num : str or int {0,...,255}, default 0
            Select which user memory to compare with

        Returns
        -------
        waveform : ndarray
            Waveform as ints spanning the resolution of the function gen or
            and empty array if waveform not in use
        """
        # Find the wavefroms in use
        waveforms_in_use = self.get_waveform_catalogue()
        if f"USER{memory_num}" in waveforms_in_use:
            # Copy the waveform to edit memory
            self.write(f"DATA:COPY EMEMory,USER{memory_num}")
            # Get the length of the waveform
            waveform_length = int(self.query("DATA:POINts? EMEMory"))
            # Get the waveform (returns binary values)
            waveform = self._inst.query_binary_values(
                "DATA:DATA? EMEMory",
                datatype="H",
                is_big_endian=True,
                container=np.ndarray,
            )
            msg = (
                f"Waveform length from native length command (DATA:POINts?) "
                f"and the processed binary values do not match, "
                f"{waveform_length} and {len(waveform)} respectively"
            )
            assert len(waveform) == waveform_length, msg
            return waveform
        print(f"Waveform USER{memory_num} is not in use")
        return np.array([])

    def set_custom_waveform(
        self,
        waveform: np.ndarray,
        normalise: bool = True,
        memory_num: int = 0,
        verify: bool = True,
        print_progress: bool = True,
    ):
        """Transfer waveform data to edit memory and then user memory.
        NOTE: Will overwrite without warnings

        Parameters
        ----------
        waveform : ndarray
            Either unnormalised arbitrary waveform (then use `normalise=True`),
            or ints spanning the resolution of the function generator
        normalise : bool
            Choose whether to normalise the waveform to ints over the
            resolution span of the function generator
        memory_num : str or int {0,...,255}, default 0
            Select which user memory to copy to
        verify : bool, default `True`
            Verify that the waveform has been transferred and is what was sent
        print_progress : bool, default `True`

        Returns
        -------
        waveform : ndarray
            The normalised waveform transferred

        Raises
        ------
        ValueError
            If the waveform is not within the permitted length or value range
        RuntimeError
            If the waveform transferred to the instrument is of a different
            length than the waveform supplied
        """
        if not 0 <= memory_num <= self._max_waveform_memory_user_locations:
            raise ValueError(
                f"The memory location {memory_num} is not a valid "
                "memory location for this model"
            )
        # Check if waveform data is suitable
        if print_progress:
            print("Check if waveform data is suitable..", end=" ")
        self._check_arb_waveform_length(waveform)
        try:
            self._check_arb_waveform_type_and_range(waveform)
        except ValueError as err:
            if print_progress:
                print(f"\n  {err}")
                print("Trying again normalising the waveform..", end=" ")
            waveform = self._normalise_to_waveform(waveform)
        if print_progress:
            print("ok")
            print("Transfer waveform to function generator..", end=" ")
        # Transfer waveform
        self._inst.write_binary_values(
            "DATA:DATA EMEMory,", waveform, datatype="H", is_big_endian=True
        )
        # Check for errors and check lengths are matching
        transfer_error = self.get_error()
        emem_wf_length = self.query("DATA:POINts? EMEMory")
        if emem_wf_length == "" or not int(emem_wf_length) == len(waveform):
            msg = (
                f"Waveform in temporary EMEMory has a length of {emem_wf_length}"
                f", not of the same length as the waveform ({len(waveform)})."
                f"\nError from the instrument: {transfer_error}"
            )
            raise RuntimeError(msg)
        if print_progress:
            print("ok")
            print(f"Copy waveform to USER{memory_num}..", end=" ")
        self.write(f"DATA:COPY USER{memory_num},EMEMory")
        if print_progress:
            print("ok")
        if verify:
            if print_progress:
                print(f"Verify waveform USER{memory_num}..")
            if f"USER{memory_num}" in self.get_waveform_catalogue():
                verif = self._verify_waveform(
                    waveform,
                    memory_num,
                    normalise=normalise,
                    print_result=print_progress,
                )
                if not verif[0]:
                    raise RuntimeError(
                        f"USER{memory_num} does not contain the waveform"
                    )
            else:
                raise RuntimeError(f"USER{memory_num} is empty")
        return waveform

    def _normalise_to_waveform(self, shape: np.ndarray) -> np.ndarray:
        """Normalise a shape of any discretisation and range to a waveform that
        can be transmitted to the function generator

        .. note::
            If you are transferring a flat/constant waveform, do not use this
            normaisation function. Transfer a waveform like
            `int(self._arbitrary_waveform_resolution/2)*np.ones(2).astype(np.int32)`
            without normalising for a well behaved flat function.

        Parameters
        ----------
        shape : array_like
            Array to be transformed to waveform, can be ints or floats,
            any normalisation or discretisation

        Returns
        -------
        waveform : ndarray
            Waveform as ints spanning the resolution of the function gen
        """
        # Check if waveform data is suitable
        self._check_arb_waveform_length(shape)
        # Normalise
        waveform = shape - np.min(shape)
        normalisation_factor = np.max(waveform)
        waveform = waveform / normalisation_factor * self._arbitrary_waveform_resolution
        return waveform.astype(np.uint16)

    def _verify_waveform(
        self,
        waveform: np.ndarray,
        memory_num: int,
        normalise: bool = True,
        print_result: bool = True,
    ) -> Tuple[bool, np.ndarray, list]:
        """Compare a waveform in user memory to argument waveform

        Parameters
        ----------
        waveform : ndarray
            Waveform as ints spanning the resolution of the function gen
        memory_num : str or int {0,...,255}, default 0
            Select which user memory to compare with
        normalise : bool, default `True`
            Normalise test waveform

        Returns
        -------
        bool
            Boolean according to equal/not equal
        instrument_waveform
            The waveform on the instrument
        list or `None`
            List of the indices where the waveforms are not equal or `None` if
            the waveforms were of different lengths
        """
        if normalise:  # make sure test waveform is normalised
            waveform = self._normalise_to_waveform(waveform)
        # Get the waveform on the instrument
        instrument_waveform = self.get_custom_waveform(memory_num)
        # Compare lengths
        len_inst_wav, len_wav = len(instrument_waveform), len(waveform)
        if not len_inst_wav == len_wav:
            if print_result:
                print(
                    f"The waveform in USER{memory_num} and the compared "
                    f"waveform are not of same length (instrument "
                    f"{len_inst_wav} vs {len_wav})"
                )
            return False, instrument_waveform, None
        # Compare each element
        not_equal = []
        for i in range(len_wav):
            if not instrument_waveform[i] == waveform[i]:
                not_equal.append(i)
        # Return depending of whether list is empty or not
        if not not_equal:  # if list is empty
            if print_result:
                print(
                    f"The waveform in USER{memory_num} and the compared "
                    f"waveform are equal"
                )
            return True, instrument_waveform, not_equal
        if print_result:
            print(
                f"The waveform in USER{memory_num} and the compared "
                f"waveform are NOT equal"
            )
        return False, instrument_waveform, not_equal

    def _check_arb_waveform_length(self, waveform: np.ndarray):
        """Checks if waveform is within the acceptable length

        Parameters
        ----------
        waveform : array_like
            Waveform or voltage list to be checked

        Raises
        ------
        ValueError
            If the waveform is not within the permitted length
        """
        if (len(waveform) < self._arbitrary_waveform_length[0]) or (
            len(waveform) > self._arbitrary_waveform_length[1]
        ):
            msg = (
                "The waveform is of length {}, which is not within the "
                "acceptable length {} < len < {}"
                "".format(len(waveform), *self._arbitrary_waveform_length)
            )
            raise ValueError(msg)

    def _check_arb_waveform_type_and_range(self, waveform: np.ndarray):
        """Checks if waveform is of int/np.int32 type and within the resolution
        of the function generator

        Parameters
        ----------
        waveform : array_like
            Waveform or voltage list to be checked

        Raises
        ------
        ValueError
            If the waveform values are not int, np.uint16 or np.int32, or the
            values are not within the permitted range
        """
        for value in waveform:
            if not isinstance(value, (int, np.uint16, np.int32)):
                raise ValueError(
                    "The waveform contains values that are not"
                    "int, np.uint16 or np.int32"
                )
            if (value < 0) or (value > self._arbitrary_waveform_resolution):
                raise ValueError(
                    f"The waveform contains values out of range "
                    f"({value} is not within the resolution "
                    f"[0, {self._arbitrary_waveform_resolution}])"
                )


## ~~~~~~~~~~~~~~~~~~~~~~~ CHANNEL CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


class FuncGenChannel:
    """Class for controlling a channel on a function generator object

    Parameters
    ----------
    fgen : `FuncGen`
        The function generator object
    channel : {1, 2}
        The channel to be controlled
    impedance : {"50ohm", "highZ"}
        Determines voltage limits associated with high impedance (whether the
        instrument is using 50ohm or high Z cannot be controlled through VISA)

    Attributes
    ----------
    _fgen : `FuncGen`
        The function generator object for which the channel exists
    _channel : {1, 2}
        The number of the channel this object is addressing
    _source : str
        "SOURce{i}:" where {i} is the channel number
    """

    _state_to_str = {"1": "ON", "0": "OFF", 1: "ON", 0: "OFF"}
    """Dictionary for converting output states to "ON" and "OFF" """

    def __init__(self, fgen: FuncGen, channel: int, impedance: str):
        self._fgen = fgen
        self._channel = channel
        self._source = f"SOURce{channel}:"
        self.impedance = impedance
        """{"50ohm", "highZ"}: Determines voltage limits associated with high
        impedance (whether the instrument is using 50ohm or high Z cannot be
        controlled through VISA)"""
        # Adopt limits dictionary from instrument
        self.channel_limits = copy.deepcopy(self._fgen.instrument_limits)
        """Channel limits for the individual channel, same form as
        `FuncGen.instrument_limits`"""

    def _impedance_dependent_limit(self, limit_type: str) -> bool:
        """Check if the limit type is impedance dependent (voltages) or
        not (frequency)

        Returns
        -------
        bool
            `True` if the limit is impedance dependent
        """
        try:  # to access the key "min" to check if impedance must be selected
            _ = self.channel_limits[limit_type][0]["min"]
            return False
        except KeyError:  # if the key does not exist
            # The impedance must be selected
            return True

    def set_stricter_limits(self):
        """Set limits for the voltage and frequency limits of the channel output
        through a series of prompts"""
        print(f"Set stricter voltage and frequency limits for channel {self._channel}")
        print("Use enter only to leave a limit unchanged.")
        # Go through the different limits in the instrument_limits dict
        for limit_type, (inst_limit_dict, unit) in self._fgen.instrument_limits.items():
            use_impedance = self._impedance_dependent_limit(limit_type)
            print(f"Set {limit_type} in {unit}", end=" ")
            if use_impedance:
                inst_limit_dict = inst_limit_dict[self.impedance]
                print(f"[{self.impedance} impedance limit]")
            else:
                print("")  # get new line
            # Go through the min and max for the limit type
            for key, inst_value in inst_limit_dict.items():
                # prompt for new value
                new_value = input(f"  {key} (instrument limit {inst_value}{unit}): ")
                if new_value == "":
                    # Do not change if empty
                    print("\tLimit not changed")
                else:
                    try:  # to convert to float
                        new_value = float(new_value)
                    except ValueError:
                        print(
                            f"\tLimit unchanged: Could not convert '{new_value}' "
                            f"to float"
                        )
                        continue  # to next item in dict
                    # Set the new limit
                    self.set_limit(limit_type, key, new_value, verbose=True)

    def set_limit(
        self, limit_type: str, bound: str, new_value: float, verbose: bool = False
    ) -> bool:
        """Set a limit if the new value is within the instrument limits and are
        self consistent (max larger than min)

        Parameterers
        ------------
        limit_type : str
            The name of the limit in the channel_limits dictionary
        bound : {"min", "max"}
            Specifies if it is the max or the min limit that is to be set
        new_value : float
            The new value to be used for the limit
        verbose : bool
            Print confirmation that the limit was set or reason for why the
            limit was not set

        Returns
        -------
        bool
            `True` if new limit set, `False` otherwise
        """
        # Short hand references
        inst_limit_dict = self._fgen.instrument_limits[limit_type]
        channel_limit_dict = self.channel_limits[limit_type]
        # Find the instrument limit and unit
        use_impedance = self._impedance_dependent_limit(limit_type)
        if use_impedance:
            inst_value = inst_limit_dict[0][self.impedance][bound]
        else:
            inst_value = inst_limit_dict[0][bound]
        unit = inst_limit_dict[1]
        # Check that the new value is within the intrument limits
        acceptable_min = bound == "min" and new_value > inst_value
        if use_impedance:
            current_min = channel_limit_dict[0][self.impedance]["min"]
        else:
            current_min = channel_limit_dict[0]["min"]
        larger_than_min = new_value > current_min
        acceptable_max = bound == "max" and new_value < inst_value and larger_than_min
        if acceptable_min or acceptable_max:  # within the limits
            # Set the new channel_limit, using the impedance depending on the
            # limit type. Beware that the shorthand cannot be used, as this
            # only changes the shorthand not the dictionary itself
            if use_impedance:
                self.channel_limits[limit_type][0][self.impedance][bound] = new_value
            else:
                self.channel_limits[limit_type][0][bound] = new_value
            if verbose:
                print(f"\tNew limit set {new_value}{unit}")
            return True
        if verbose:  # print description of why the limit was not set
            if larger_than_min:
                reason = "larger" if bound == "max" else "smaller"
                print(
                    f"\tNew limit NOT set: {new_value}{unit} is {reason} than "
                    f"the instrument limit ({inst_value}{unit})"
                )
            else:
                print(
                    f"\tNew limit NOT set: {new_value}{unit} is smaller than the "
                    f"current set minimum ({current_min}{unit})"
                )
        return False

    # Get currently used parameters from function generator
    def get_output_state(self) -> int:
        """Returns 0 for "OFF", 1 for "ON" """
        return int(self._fgen.query(f"OUTPut{self._channel}:STATe?"))

    def get_function(self) -> str:
        """Returns string of function name"""
        return self._fgen.query(f"{self._source}FUNCtion:SHAPe?")

    def get_amplitude(self) -> float:
        """Returns peak-to-peak voltage in volts"""
        return float(self._fgen.query(f"{self._source}VOLTage:AMPLitude?"))

    def get_offset(self) -> float:
        """Returns offset voltage in volts"""
        return float(self._fgen.query(f"{self._source}VOLTage:OFFSet?"))

    def get_frequency(self) -> float:
        """Returns frequency in Hertz"""
        return float(self._fgen.query(f"{self._source}FREQuency?"))

    # Get limits set in the channel class
    def get_frequency_lims(self) -> List[float]:
        """Returns list of min and max frequency limits"""
        return [self.channel_limits["frequency lims"][0][key] for key in ["min", "max"]]

    def get_voltage_lims(self) -> List[float]:
        """Returns list of min and max voltage limits for the current impedance"""
        return [
            self.channel_limits["voltage lims"][0][self.impedance][key]
            for key in ["min", "max"]
        ]

    def get_amplitude_lims(self) -> List[float]:
        """Returns list of min and max amplitude limits for the current impedance"""
        return [
            self.channel_limits["amplitude lims"][0][self.impedance][key]
            for key in ["min", "max"]
        ]

    def get_settings(self) -> dict:
        """Get the settings for the channel

        Returns
        -------
        current_settings : dict
            Settings currently in use as a dictionary with keys output,
            function, amplitude, offset, and frequency and values tuples of
            the corresponding return and unit
        """
        return {
            "output": (self._state_to_str[self.get_output_state()], ""),
            "function": (self.get_function(), ""),
            "amplitude": (self.get_amplitude(), "Vpp"),
            "offset": (self.get_offset(), "V"),
            "frequency": (self.get_frequency(), "Hz"),
        }

    def print_settings(self):
        """Print the settings currently in use for the channel (Recommended
        to use the `FuncGen.print_settings` for printing both channels)
        """
        settings = self.get_settings()
        longest_key = max([len(key) for key in settings.keys()])
        print("\nCurrent settings for channel {}".format(self._channel))
        print("==============================")
        for key, (val, unit) in settings.items():
            print("{:>{num_char}s} {} {}".format(key, val, unit, num_char=longest_key))

    def set_settings(self, settings: dict):
        """Set the settings of the channel with a settings dictionary. Will
        set the outout to OFF before applyign the settings (and turn the
        channel ON or leave it OFF depending on the settings dict)

        Parameteres
        -----------
        settings : dict
            Settings dictionary as returned by `get_settings`: should have
            keys output, function, amplitude, offset, and frequency
        """
        # First turn off to ensure no potentially harmful
        # combination of settings
        self.set_output_state("OFF")
        # Set settings according to dictionary
        self.set_function(settings["function"][0])
        self.set_amplitude(settings["amplitude"][0])
        self.set_offset(settings["offset"][0])
        self.set_frequency(settings["frequency"][0])
        self.set_output_state(settings["output"][0])

    def set_output_state(self, state: Union[int, str]):
        """Enables or diables the output of the channel

        Parameters
        ----------
        state : int or str
            "ON" or int 1 to enable
            "OFF" or int 0 to disable

        Raises
        ------
        NotSetError
            If `self._fgen.verify_param_set` is `True` and the value after
            applying the set function does not match the value returned by the
            get function
        """
        err_msg = f"turn channel {self._channel} to state {state}"
        self._fgen.write(
            f"OUTPut{self._channel}:STATe {state}", custom_err_message=err_msg
        )
        if self._fgen.verify_param_set:
            actual_state = self.get_output_state()
            if not actual_state == state:
                msg = (
                    f"Channel {self._channel} was not turned {state}, it is "
                    f"{self._state_to_str[actual_state]}.\n"
                    f"Error from the instrument: {self._fgen.get_error()}"
                )
                raise NotSetError(msg)

    def get_output(self) -> int:
        """Wrapper for get_output_state"""
        return self.get_output_state()

    def set_output(self, state: Union[int, str]):
        """Wrapper for set_output_state"""
        self.set_output_state(state)

    def set_function(self, shape: str):
        """Set the function shape of the output

        Parameters
        ----------
        shape : {SINusoid, SQUare, PULSe, RAMP, PRNoise, <Built_in>, USER[0],
                 USER1, ..., USER255, EMEMory, EFILe}
            <Built_in>::={StairDown|StairUp|Stair Up&Dwn|Trapezoid|RoundHalf|
            AbsSine|AbsHalfSine|ClippedSine|ChoppedSine|NegRamp|OscRise|
            OscDecay|CodedPulse|PosPulse|NegPulse|ExpRise|ExpDecay|Sinc|
            Tan|Cotan|SquareRoot|X^2|HaverSine|Lorentz|Ln(x)|X^3|CauchyDistr|
            BesselJ|BesselY|ErrorFunc|Airy|Rectangle|Gauss|Hamming|Hanning|
            Bartlett|Blackman|Laylight|Triangle|DC|Heart|Round|Chirp|Rhombus|
            Cardiac}

        Raises
        ------
        NotSetError
            If `self._fgen.verify_param_set` is `True` and the value after
            applying the set function does not match the value returned by the
            get function
        """
        cmd = f"{self._source}FUNCtion:SHAPe {shape}"
        self._fgen.write(cmd, custom_err_message=f"set function {shape}")
        if self._fgen.verify_param_set:
            actual_shape = self.get_function()
            if not actual_shape == shape:
                msg = (
                    f"Function {shape} was not set on channel {self._channel}, "
                    f"it is {actual_shape}. Check that the function name is "
                    f"correctly spelt. Look up `set_function.__doc__` to see "
                    f"available shapes.\n Error from the instrument: "
                    f"{self._fgen.get_error()}"
                )
                raise NotSetError(msg)

    def set_amplitude(self, amplitude: float):
        """Set the peak-to-peak amplitude in volts

        Parameters
        ----------
        amplitude : float or {"max", "min"}
            0.1mV or four digits resolution, "max" or "min" will set the
            amplitude to the maximum or minimum limit given in `channel_limits`

        Raises
        ------
        NotSetError
            If `self._fgen.verify_param_set` is `True` and the value after
            applying the set function does not match the value returned by the
            get function
        """
        # Check if keyword min or max is given
        if str(amplitude).lower() in ["min", "max"]:
            unit = ""  # no unit for MIN/MAX
            # Look up what the limit is for this keyword
            amplitude = self.channel_limits["amplitude lims"][0][self.impedance][
                str(amplitude).lower()
            ]
        else:
            unit = "Vpp"
            # Check if the given amplitude is within the current limits
            min_ampl, max_ampl = self.get_amplitude_lims()
            if amplitude < min_ampl or amplitude > max_ampl:
                msg = (
                    f"Could not set the amplitude {amplitude}{unit} as it "
                    f"is not within the amplitude limits set for the instrument "
                    f"[{min_ampl}, {max_ampl}]{unit}"
                )
                raise NotSetError(msg)
        # Check that the new amplitude will not violate voltage limits
        min_volt, max_volt = self.get_voltage_lims()
        current_offset = self.get_offset()
        if (
            amplitude / 2 - current_offset < min_volt
            or amplitude / 2 + current_offset > max_volt
        ):
            msg = (
                f"Could not set the amplitude {amplitude}{unit} as the amplitude "
                f"combined with the offset ({current_offset}V) will be outside the "
                f"absolute voltage limits [{min_volt}, {max_volt}]{unit}"
            )
            raise NotSetError(msg)
        # Set the amplitude
        cmd = f"{self._source}VOLTage:LEVel {amplitude}{unit}"
        err_msg = f"set amplitude {amplitude}{unit}"
        self._fgen.write(cmd, custom_err_message=err_msg)
        # Verify that the amplitude has been set
        if self._fgen.verify_param_set:
            actual_amplitude = self.get_amplitude()
            # Multiply with the appropriate factor according to SI prefix, or
            # if string is empty, use the value looked up from channel_limits earlier
            if not unit == "":
                check_amplitude = amplitude * _SI_prefix_to_factor(unit)
            else:
                check_amplitude = amplitude
            if not actual_amplitude == check_amplitude:
                msg = (
                    f"Amplitude {amplitude}{unit} was not set on channel "
                    f"{self._channel}, it is {actual_amplitude}Vpp. Check that "
                    f" the number is within the possible range and in the "
                    f"correct format.\nError from the instrument: "
                    f"{self._fgen.get_error()}"
                )
                raise NotSetError(msg)

    def set_offset(self, offset: float, unit: str = "V"):
        """Set offset in volts (or mV, see options)

        Parameters
        ----------
        offset : float
            Unknown resolution, guessing 0.1mV or four digits resolution
        unit : {mV, V}, default V

        Raises
        ------
        NotSetError
            If `self._fgen.verify_param_set` is `True` and the value after
            applying the set function does not match the value returned by the
            get function
        """
        # Check that the new offset will not violate voltage limits
        min_volt, max_volt = self.get_voltage_lims()
        current_amplitude = self.get_amplitude()
        offset = _SI_prefix_to_factor(unit) * offset
        if (
            current_amplitude / 2 - offset < min_volt
            or current_amplitude / 2 + offset > max_volt
        ):
            msg = (
                f"Could not set the offset {offset}V as the offset combined "
                f"with the amplitude ({current_amplitude}V) will be outside "
                f"the absolute voltage limits [{min_volt}, {max_volt}]V"
            )
            raise NotSetError(msg)
        # Set the offset
        cmd = f"{self._source}VOLTage:LEVel:OFFSet {offset}{unit}"
        err_msg = f"set offset {offset}{unit}"
        self._fgen.write(cmd, custom_err_message=err_msg)
        # Verify that the offset has been set
        if self._fgen.verify_param_set:
            actual_offset = self.get_offset()
            # Multiply with the appropriate factor according to SI prefix
            check_offset = offset * _SI_prefix_to_factor(unit)
            if not actual_offset == check_offset:
                msg = (
                    f"Offset {offset}{unit} was not set on channel "
                    f"{self._channel}, it is {actual_offset}V. Check that the "
                    f"number is within the possible range and in the correct "
                    f"format.\nError from the instrument: {self._fgen.get_error()}"
                )
                raise NotSetError(msg)

    def set_frequency(self, freq: float, unit: str = "Hz"):
        """Set the frequency in Hertz (or mHz, kHz, MHz, see options)

        Parameters
        ----------
        freq : float
            The resolution is 1 μHz or 12 digits.
        unit : {mHz, Hz, kHz, MHz}, default Hz

        Raises
        ------
        NotSetError
            If `self._fgen.verify_param_set` is `True` and the value after
            applying the set function does not match the value returned by the
            get function
        """
        if str(freq).lower() in ["min", "max"]:  # handle min and max keywords
            unit = ""  # no unit for MIN/MAX
            # Look up what the limit is for this keyword
            freq = self.channel_limits["frequency lims"][0][str(freq).lower()]
        else:
            # Check if the given frequency is within the current limits
            min_freq, max_freq = self.get_frequency_lims()
            freq = _SI_prefix_to_factor(unit) * freq
            if freq < min_freq or freq > max_freq:
                msg = (
                    f"Could not set the frequency {freq}Hz as it is not "
                    f"within the frequency limits set for the instrument "
                    f"[{min_freq}, {max_freq}]Hz"
                )
                raise NotSetError(msg)
        # Set the frequency
        self._fgen.write(
            f"{self._source}FREQuency:FIXed {freq}{unit}",
            custom_err_message=f"set frequency {freq}{unit}",
        )
        # Verify that the amplitude has been set
        if self._fgen.verify_param_set:
            actual_freq = self.get_frequency()
            # Multiply with the appropriate factor according to SI prefix, or
            # if string is empty, use the value looked up from channel_limits earlier
            if not unit == "":
                check_freq = freq * _SI_prefix_to_factor(unit)
            else:
                check_freq = freq
            if not actual_freq == check_freq:
                msg = (
                    f"Frequency {freq}{unit} was not set on channel {self._channel}"
                    f", it is {actual_freq}Hz. Check that the number is within "
                    f"the possible range and in the correct format.\nError "
                    f"from the instrument: {self._fgen.get_error()}"
                )
                raise NotSetError(msg)


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EXAMPLES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


def example_basic_control(address: str):
    """Example showing how to connect, and the most basic control of the
    instrument parameters"""
    print("\n\n", example_basic_control.__doc__)
    with FuncGen(address) as fgen:
        fgen.ch1.set_function("SIN")
        fgen.ch1.set_frequency(25, unit="Hz")
        fgen.ch1.set_offset(50, unit="mV")
        fgen.ch1.set_amplitude(0.002)
        fgen.ch1.set_output("ON")
        fgen.ch2.set_output("OFF")
        # Alternatively fgen.ch1.print_settings() to show from one channel only
        fgen.print_settings()


def example_change_settings(address: str):
    """Example showing how to get the current settings of the instrument,
    store them, change a setting and then restore the initial settings"""
    print("\n\n", example_change_settings.__doc__)
    with FuncGen(address) as fgen:
        fgen.print_settings()
        print("Saving these settings..")
        settings = fgen.get_settings()
        print("Change to 1Vpp amplitude for channel 1..")
        fgen.ch1.set_amplitude(1)
        fgen.print_settings()
        print("Reset back to initial settings..")
        fgen.set_settings(settings)
        fgen.print_settings()


def example_lock_frequencies(address: str):
    """Example showing the frequency being set to 10Hz and then the frequency
    lock enabled, using the frequency at ch1 as the common frequency"""
    print("\n\n", example_lock_frequencies.__doc__)
    with FuncGen(address, verbose=False) as fgen:
        fgen.ch1.set_frequency(10)
        fgen.set_frequency_lock("ON", use_channel=1)


def example_changing_limits(address: str):
    """Example showing how limits can be read and changed"""
    print("\n\n", example_changing_limits.__doc__)
    with FuncGen(address) as fgen:
        lims = fgen.ch1.get_frequency_lims()
        print("Channel 1 frequency limits: {}".format(lims))
        print("Change the lower limit to 2Hz..")
        fgen.ch1.set_limit("frequency lims", "min", 2)
        lims = fgen.ch1.get_frequency_lims()
        print("Channel 1 frequency limits: {}".format(lims))
        print("Try to set ch1 frequency to 1Hz..")
        try:
            fgen.ch1.set_frequency(1)
        except NotSetError as err:
            print(err)


def example_set_and_use_custom_waveform(
    fgen: FuncGen = None,
    address: str = None,
    channel: int = 1,
    plot_signal: bool = True,
):
    """Example showing a waveform being created, transferred to the instrument,
    and applied to a channel"""
    print("\n\n", example_set_and_use_custom_waveform.__doc__)
    # Create a signal
    x = np.linspace(0, 4 * np.pi, 8000)
    signal = np.sin(x) + x / 5
    if plot_signal:  # plot the signal for visual control
        import matplotlib.pyplot as plt

        plt.plot(signal)
        plt.show()
    # Create initialise fgen if it was not supplied
    if fgen is None:
        fgen = FuncGen(address)
        close_fgen = True  # specify that it should be closed at end of function
    else:
        close_fgen = False  # do not close the supplied fgen at end
    print("Current waveform catalogue")
    for i, wav in enumerate(fgen.get_waveform_catalogue()):
        print(f"  {i}: {wav}")
    # Transfer the waveform
    fgen.set_custom_waveform(signal, memory_num=5, verify=True)
    print("New waveform catalogue:")
    for i, wav in enumerate(fgen.get_waveform_catalogue()):
        print(f"  {i}: {wav}")
    print(f"Set new wavefrom to channel {channel}..", end=" ")
    fgen.channels[channel - 1].set_output_state("OFF")
    fgen.channels[channel - 1].set_function("USER5")
    print("ok")
    # Print current settings
    fgen.get_settings()
    if close_fgen:
        fgen.close()


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN FUNCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

if __name__ == "__main__":
    example_basic_control(_VISA_ADDRESS)
    example_change_settings(_VISA_ADDRESS)
    example_lock_frequencies(_VISA_ADDRESS)
    example_changing_limits(_VISA_ADDRESS)
    with FuncGen(_VISA_ADDRESS) as fgen:
        example_set_and_use_custom_waveform(fgen)
