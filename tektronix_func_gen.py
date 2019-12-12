# -*- coding: utf-8 -*-
"""
Control of Tektronix AFG1022 function generator through PyVISA

To build the documentation use [pdoc3](https://pdoc3.github.io/pdoc/) and run `$ pdoc --html tektronix_func_gen`

Andreas Svela 2019

Todo:
    * add AM/FM capabilites
"""

import numpy as np
import pyvisa


## ~~~~~~~~~~~~~~~~~~~~~ FUNCTION GENERATOR CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

class func_gen():
    """Class for interacting with Tektronix function generator

    Parameters
    ----------
    address : str
        VISA address of insrument
    impedance : tuple of {"highZ", "50ohm"}, default ("highZ",)*2
        Determines voltage limits associated with high impedance (whether the
        instrument is using 50ohm or high Z cannot be controlled through VISA).
        For example `("highZ", "50ohm")` for to use high Z for ch1 and
        50 ohm for ch2
    timeout : int, default 5000
        Timeout in milliseconds of instrument connection
    verify_param_set : bool, default False
        Verify that a value is successfully set after executing a set function
    verbose : bool, default `True`
        Choose whether to print information such as model upon connecting etc
    override_compatibility : bool, default `False`
        If the instrument limits for the model connected to are not known
        `NotCompatibleError` will be raised. To override and use AFG1022 limits,
        set to `True`. Note that this might lead to unexpected behaviour for
        custom waveforms and 'MIN'/'MAX' keywords.

    Attributes
    ----------
    address : str
        The VISA address of the instrument
    id : str
        Comma separated string with maker, model, serial and firmware of the instrument
    inst : `pyvisa.resources.Resource`
        The PyVISA resource
    channels : tuple of func_gen_channel
        Objects to control the channels
    ch1 : func_gen_channel
        Short hand for `channels[0]` Object to control channel 1
    ch2 : func_gen_channel
        Short hand for `channels[1]` Object to control channel 2
    instrument_limits : dict
        Contains the following keys with subdictionaries
        `frequency lims`
            Containing the frequency limits for the instrument where the keys "min"
            and "max" have values corresponding to minimum and maximum frequencies in Hz
        `voltage lims`
            Contains the maximum absolute voltage the instrument can output for the keys
            "50ohm" and "highZ" according to the impedance setting
        `amplitude lims`
            Contains the smallest and largest possible amplitudes where the keys
            "50ohm" and "highZ" will have subdictionaries with keys "min" and "max"
    arbitrary_waveform_length : list
        The permitted minimum and maximum length of an arbitrary waveform, e.g. [2, 8192]
    arbitrary_waveform_resolution : int
        The vertical resolution of the arbitrary waveform, for instance 14 bit
        => 2**14-1 = 16383

    Raises
    ------
    pyvisa.Error
        If the supplied VISA address cannot be connected to
    NotCompatibleError
        If the instrument limits for the model connected to are not known (Call the class with `override_compatibility=True` to override and use AFG1022 limits)
    """

    def __init__(self, address, impedance=("highZ",)*2, timeout=5000,
                 verify_param_set=False, override_compatibility=False, verbose=True):
        self.override_compatibility = override_compatibility
        self.address = address
        self.timeout = timeout
        self.verify_param_set = verify_param_set
        self.verbose = verbose
        try:
            rm = pyvisa.ResourceManager()
            self.inst = rm.open_resource(address)
        except pyvisa.Error:
            print("\nVisaError: Could not connect to \'{}\'".format(address))
            raise
        self.inst.timeout = self.timeout
        # Clear all the event registers and queues used in the instrument status and event reporting system
        self.write("*CLS")
        # Get information about the connected device
        self.id = self.query("*IDN?") # get the id of the connected device
        if self.id == '': # second query might be needed due to unknown reason (suspecting *CLS does something weird)
            self.id = self.query("*IDN?")
        self.maker, self.model, self.serial = self.id.split(",")[:3]
        if self.verbose: print("Connected to {} model {}, serial {}".format(self.maker, self.model, self.serial))
        self.initialise_model_properties()
        self.channels = self.spawn_channel(1, impedance[0]), self.spawn_channel(2, impedance[1])
        self.ch1, self.ch2 = self.channels

    # __enter__ and __exit__ functions defined to use the module with a 'with' statement
    # with func_gen(verify_param_set=True) as fgen:
    #    fgen.get_settings()
    # This gives automatic handling of closing down the connection when fgen goes out of scope
    # The kwargs in __enter__ will be passed on to __init__
    def __enter__(self, **kwargs):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Close the connection to the instrument"""
        self.inst.close()

    def initialise_model_properties(self):
        """Initialises the limits of what the instrument can handle according
        to the instrument model

        Raises
        ------
        NotCompatibleError
            If the connected model is not necessarily compatible with this package,
            limits are not known.
        """
        if self.model == 'AFG1022' or self.override_compatibility:
            self.instrument_limits = {
                    "frequency lims": ({"min": 1e-6, "max": 25e6}, "Hz"),
                    "voltage lims":   ({"50ohm": {"min": -5, "max": 5},
                                        "highZ": {"min": -10, "max": 10}}, "V"),
                    "amplitude lims": ({"50ohm": {"min": 0.001, "max": 10},
                                        "highZ": {"min": 0.002, "max": 20}}, "Vpp")}
            self.arbitrary_waveform_length = [2, 8192] # min length, max length
            self.arbitrary_waveform_resolution = 16383 # 14 bit
        else:
            raise NotCompatibleError("Model {} not supported, no limits set!\n\tTo use the limits for AFG1022, call the class with 'override_compatibility=True'\n\tNote that this might lead to unexpected behaviour for custom waveforms and 'MIN'/'MAX' keywords.")

    def write(self, command, custom_err_message=None):
        """Write a VISA command to the instrument

        Parameters
        ----------
        command : str
            The VISA command to be written to the instrument
        custom_err_message : str, default `None`
            When `None`, the RuntimeError message is "Writing command {command} failed: pyvisa returned StatusCode ..".
            Otherwise, if a message is supplied "Could not {message}: pyvisa returned StatusCode .."

        Returns
        -------
        bytes : int
            Number of bytes tranferred
        status : pyvisa.constants.StatusCode
            Return value of the library call

        Raises
        ------
        RuntimeError
            If status returned by PyVISA write command is not `pyvisa.constants.StatusCode.success`
        """
        bytes, status = self.inst.write(command)
        if not status == pyvisa.constants.StatusCode.success:
            if custom_err_message is not None:
                raise RuntimeError("Could not {}: pyvisa returned StatusCode {} ({})".format(custom_err_message, status, str(status)))
            else:
                raise RuntimeError("Writing command {} failed: pyvisa returned StatusCode {} ({})".format(command, status, str(status)))
        return bytes, status

    def query(self, command):
        """Query the instrument

        Parameters
        ----------
        command : str
            The VISA query command

        Returns
        -------
        str
            The instrument's response
        """
        return self.inst.query(command).strip()

    def get_error(self):
        """Get the contents of the Error/Event queue on the device

        Returns
        -------
        str
            Error/event number, description of error/event
        """
        return self.query("SYSTEM:ERROR:NEXT?")

    def spawn_channel(self, channel, impedance):
        """Wrapper function to create a `func_gen_channel` object for
        a channel -- see the class docstring"""
        return func_gen_channel(self, channel, impedance)

    def get_settings(self):
        """Get dictionaries of the current settings of the two channels

        Parameters
        ----------
        print_settings : bool, default `True`
            `True` prints table of the current setting for both channels

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
        # find the necessary padding for the table columns
        # by evaluating the maximum length of the entries
        key_padding = max([len(key) for key in settings[0].keys()])
        ch_paddings = [max([len(str(val[0])) for val in ch_settings.values()]) for ch_settings in settings]
        padding = [key_padding]+ch_paddings
        print("\nCurrent settings for {} {} {}\n".format(self.maker, self.model, self.serial))

        row_format = "{:>{padd[0]}s} {:{padd[1]}s} {:{padd[2]}s} {}"
        table_header = row_format.format("Setting", "Ch1", "Ch2", "Unit", padd=padding)
        print(table_header)
        print("="*len(table_header))
        for (ch1key, (ch1val, unit)), (ch2key, (ch2val, _)) in zip(settings[0].items(), settings[1].items()):
            print(row_format.format(ch1key, str(ch1val), str(ch2val), unit, padd=padding))

    def syncronise_waveforms(self):
        """Syncronise waveforms of the two channels when using the same frequency

        Note: Does NOT enable the frequency lock that can be enabled on the user
        interface of the instrument)
        """
        self.write(":PHAS:INIT", custom_err_message="syncronise waveforms")

    def get_frequency_lock(self):
        """Check if frequency lock is enabled

        Returns
        -------
        bool
            `True` if frequency lock enabled
        """
        # if one is locked so is the other, so just need to check one
        return int(self.query("SOURCE1:FREQuency:CONCurrent?")) == 1

    def enable_frequency_lock(self, use_channel=1):
        """Enable the frequency lock to make the two channels have the same frequency
        and phase of their signals, also after adjustments.

        See also `func_gen.syncronise_waveforms` for one-time sync only.

        Parameters
        ----------
        use_channel : int, default 1
            The channel whose frequency shall be used as the common freqency
        """
        self.write("SOURCE{}:FREQuency:CONCurrent ON".format(use_channel),
                   custom_err_message="enable frequency lock")

    def disable_frequency_lock(self):
        """Disable the frequency lock that makes the two channels have the same
        frequency and phase of their signals, also after adjustments
        """
        if self.get_frequency_lock():
            # sufficient to disable for only one of the channels
            self.write("SOURCE1:FREQuency:CONCurrent OFF")
        else:
            if self.verbose: print("(!) {}: Tried to disable frequency lock, but frequency lock was not enabled".format(self.model))

    def software_trig(self):
        """NOT TESTED: sends a trigger signal to the device (for bursts or modulations)"""
        self.write("*TRG", custom_err_message="send trigger signal")


    ## ~~~~~~~~~~~~~~~~~~~~~ CUSTOM WAVEFORM FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~ ##

    def get_waveform_catalogue(self):
        """Get list of the waveforms that are in use (not empty)

        Returns
        -------
        catalogue : list
            Strings with the names of the user functions that are not empty
        """
        catalogue = self.query("DATA:CATalog?").split(",")
        catalogue =  [wf[1:-1] for wf in catalogue] # strip off extra quotes
        return catalogue

    def get_custom_waveform(self, memory_num):
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
        # find the wavefroms in use
        waveforms_in_use = self.get_waveform_catalogue()
        if "USER{}".format(memory_num) in waveforms_in_use:
            # copy the waveform to edit memory
            self.write("DATA:COPY EMEMory,USER{}".format(memory_num))
            # get the length of the waveform
            waveform_length = int(self.query("DATA:POINts? EMEMory"))
            # get the waveform (returns binary values)
            waveform = self.inst.query_binary_values("DATA:DATA? EMEMory", datatype='H', is_big_endian=True, container=np.ndarray)
            assert len(waveform) == waveform_length, "Waveform length from native length command (DATA:POINts?) and the processed binary values do not match, {} and {} respectively".format(waveform_length, len(waveform))
            return waveform
        else:
            print("Waveform USER{} is not in use".format(memory_num))
            return np.array([])

    def set_custom_waveform(self, waveform, normalise=True, memory_num=0, verify=True, print_progress=True):
        """Transfer waveform data to edit memory and then user memory.
        NOTE: Will overwrite without warnings

        Parameters
        ----------
        waveform : ndarray
            Either unnormalised arbitrary waveform (then use `normalise=True`), or
            ints spanning the resolution of the function generator
        normalise : bool
            Choose whether to normalise the waveform to ints over the resolution
            span of the function generator
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
        # check if waveform data is suitable
        if print_progress: print("Check if waveform data is suitable..", end=" ")
        self.check_arb_waveform_length(waveform)
        try:
            self.check_arb_waveform_type_and_range(waveform)
        except ValueError as e:
            if print_progress: print("\n  "+str(e))
            if print_progress: print("Trying again normalising the waveform..", end=" ")
            waveform = self.normalise_to_waveform(waveform)
        if print_progress: print("ok")
        # transfer waveform
        if print_progress: print("Transfer waveform to function generator..", end=" ")
        self.inst.write_binary_values("DATA:DATA EMEMory,", waveform, datatype='H', is_big_endian=True)
        # The first query after the write_binary_values returns '', so here is a mock query
        self.query("")
        transfer_error = self.get_error()
        emem_wf_length = self.query("DATA:POINts? EMEMory")
        if emem_wf_length == '' or not int(emem_wf_length) == len(waveform):
            raise RuntimeError("Waveform in temporary EMEMory has a length of {}, not of the same length as the waveform ({}).\nError from the instrument: {}".format(emem_wf_length, len(waveform), transfer_error))
        if print_progress: print("ok")
        if print_progress: print("Copy waveform to USER{}..".format(memory_num), end=" ")
        self.write("DATA:COPY USER{},EMEMory".format(memory_num))
        if print_progress: print("ok")
        if verify:
            if print_progress: print("Verify waveform..".format(memory_num))
            if "USER{}".format(memory_num) in self.get_waveform_catalogue():
                self.verify_waveform(waveform, memory_num, normalise=normalise, print_result=print_progress)
            else:
                print("(!) USER{} is empty".format(memory_num))
        return waveform

    def normalise_to_waveform(self, shape):
        """Normalise a shape of any discretisation and range to a waveform that
        can be transmitted to the function generator

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
        # check if waveform data is suitable
        self.check_arb_waveform_length(shape)
        # normalise
        waveform = shape - np.min(shape)
        normalisation_factor = np.max(waveform)
        waveform = waveform/normalisation_factor*self.arbitrary_waveform_resolution
        return waveform.astype(np.uint16)

    def verify_waveform(self, waveform, memory_num, normalise=True, print_result=True):
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
        if normalise:# make sure test waveform is normalised
            waveform = self.normalise_to_waveform(waveform)
        # get the waveform on the instrument
        instrument_waveform = self.get_custom_waveform(memory_num)
        # compare lengths
        len_inst_wav, len_wav = len(instrument_waveform), len(waveform)
        if not len_inst_wav == len_wav:
            if print_result: print("The waveform in USER{} and the compared waveform are not of same length (instrument {} vs {})".format(memory_num, len_inst_wav, len_wav))
            return False, instrument_waveform, None
        # compare each element
        not_equal = []
        for i in range(len_wav):
            if not instrument_waveform[i] == waveform[i]:
                not_equal.append(i)
        # return depending of whether list is empty or not
        if not not_equal: # if list is empty
            if print_result: print("The waveform in USER{} and the compared waveform are equal".format(memory_num))
            return True, instrument_waveform, not_equal
        if print_result: print("The waveform in USER{} and the compared waveform are NOT equal".format(memory_num))
        return False, instrument_waveform, not_equal

    def check_arb_waveform_length(self, waveform):
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
        if (len(waveform) < self.arbitrary_waveform_length[0]) or (len(waveform) > self.arbitrary_waveform_length[1]):
            raise ValueError("The waveform is of length {}, which is not within the acceptable length {} < len < {}".format(len(waveform), *self.arbitrary_waveform_length))

    def check_arb_waveform_type_and_range(self, waveform):
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
                raise ValueError("The waveform contains values that are not int, np.uint16 or np.int32")
            if (value < 0) or (value > self.arbitrary_waveform_resolution):
                raise ValueError("The waveform contains values out of range ({} is not within the resolution [0, {}])".format(value, self.arbitrary_waveform_resolution))


## ~~~~~~~~~~~~~~~~~~~~~~~ CHANNEL CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

class func_gen_channel:
    """Class for controlling a channel on a function generator object

    Parameters
    ----------
    func_gen : `func_gen`
        The function generator object
    channel : {1, 2}
        The channel to be controlled
    impedance : {"50ohm", "highZ"}
        Determines voltage limits associated with high impedance (whether the
        instrument is using 50ohm or high Z cannot be controlled through VISA)

    Attributes
    ----------
    func_gen : `func_gen`
        The function generator object for which the channel exists
    channel : {1, 2}
        The channel number
    impedance : {"50ohm", "highZ"}
        Determines voltage limits associated with high impedance (whether the
        instrument is using 50ohm or high Z cannot be controlled through VISA)
    source : str
        "SOURce{}:" where {} is the channel number
    settings_units : list
        The units for the settings produced by `func_gen_channel.get_settings`
    state_str : dict
        For conversion from states 1 and 2 to "ON" and "OFF"
    """

    # Attributes
    state_str = {"1": "ON", "0": "OFF",
                  1 : "ON",  0 : "OFF"}
    """Dictionary for converting output states to "ON" and "OFF" """

    def __init__(self, func_gen, channel, impedance):
        self.func_gen = func_gen
        self.channel = channel
        self.source = "SOURce{}:".format(channel)
        self.impedance = impedance
        # adopt limits dictionary from instrument
        self.channel_limits = self.func_gen.instrument_limits

    def set_stricter_limits(self):
        """Set limits for the voltage and frequency limits of the channel output
        through a series of prompts"""
        print("Set stricter voltage and frequency limits for channel {}".format(self.channel))
        print("Use enter only to leave a limit unchanged.")
        # Go through the different limits in the instrument_limits dict
        for limit_type, (inst_limit_dict, unit) in self.func_gen.instrument_limits.items():
            try: # to access the key "min" to check if impedance must be selected
                _ = inst_limit_dict["min"]
                use_impedance = False
            except KeyError: # if the key does not exist
                # the impedance must be selected
                use_impedance = True
                inst_limit_dict = inst_limit_dict[self.impedance]
            print("Set {} in {}".format(limit_type, unit), end=" ")
            if use_impedance:
                print("[{} impedance limit]".format(self.impedance))
            else:
                print("") # get new line
            # Go through the min and max for the limit type
            for key, inst_value in inst_limit_dict.items():
                # prompt for new value
                new_value = input("  {} (instrument limit {}{}): ".format(key, inst_value, unit))
                if new_value == "":
                    # do not change if empty
                    print("\tLimit not changed")
                else:
                    try: # to convert to float
                        new_value = float(new_value)
                    except ValueError:
                        print("\tLimit unchanged: Could not convert {} to float".format(new_value))
                        continue # to next item in dict
                    # check that the new value is within the intrument limits
                    acceptable_min = key == "min" and new_value > inst_value
                    current_min = self.channel_limits[limit_type][0][self.impedance]["min"] if use_impedance else self.channel_limits[limit_type][0]["min"]
                    larger_than_min = new_value > current_min
                    acceptable_max = key == "max" and new_value < inst_value and larger_than_min
                    if acceptable_min or acceptable_max: # within the limits
                        # set the new channel_limit, using the impedance depending on the limit type
                        if use_impedance:
                            self.channel_limits[limit_type][0][self.impedance][key] = new_value
                        else:
                            self.channel_limits[limit_type][0][key] = new_value
                        print("\tNew limit set {}{}".format(new_value, unit))
                    else: # print description of why the limit was not set
                        if larger_than_min:
                            reason = "larger" if key == "max" else "smaller"
                            print("\tNew limit NOT set: {}{unit} is {} than the instrument limit ({}{unit})".format(new_value, reason, inst_value, unit=unit))
                        else:
                            print("\tNew limit NOT set: {}{unit} is smaller than the current set minimum ({}{unit})".format(new_value, current_min, unit=unit))

    # get currently used parameters from function generator
    def get_output_state(self):
        """Returns "0" for "OFF", "1" for "ON" """
        return self.func_gen.query("OUTPut{}:STATe?".format(self.channel))
    def get_function(self):
        """Returns string of function name"""
        return self.func_gen.query("{}FUNCtion:SHAPe?".format(self.source))
    def get_amplitude(self):
        """Returns peak-to-peak voltage in volts"""
        return float(self.func_gen.query("{}VOLTage:AMPLitude?".format(self.source)))
    def get_offset(self):
        """Returns offset voltage in volts"""
        return float(self.func_gen.query("{}VOLTage:OFFSet?".format(self.source)))
    def get_frequency(self):
        """Returns frequency in Hertz"""
        return float(self.func_gen.query("{}FREQuency?".format(self.source)))

    # get limits set in the channel class
    def get_frequency_lims(self):
        """Returns list of min and max frequency limits"""
        return [self.channel_limits["frequency lims"][0][key] for key in ["min", "max"]]
    def get_voltage_lims(self):
        """Returns list of min and max voltage limits for the current impedance"""
        return [self.channel_limits["voltage lims"][0][self.impedance][key] for key in ["min", "max"]]
    def get_amplitude_lims(self):
        """Returns list of min and max amplitude limits for the current impedance"""
        return [self.channel_limits["amplitude lims"][0][self.impedance][key] for key in ["min", "max"]]

    def get_settings(self):
        """Get the settings for the channel

        Returns
        -------
        current_settings : dict
            Settings currently in use as a dictionary with keys output, function,
            amplitude, offset, and frequency and values tuples of the corresponding
            return and unit
        """
        return {"output":    (self.state_str[self.get_output_state()], ""),
                "function":  (self.get_function(),  ""),
                "amplitude": (self.get_amplitude(), "Vpp"),
                "offset":    (self.get_offset(),    "V"),
                "frequency": (self.get_frequency(), "Hz")}

    def print_settings(self):
        """Print the settings currently in use for the channel
        (Recommended to use the `func_gen.print_settings` for printing both channels)
        """
        settings = self.get_settings()
        longest_key = max([len(key) for key in settings.keys()])
        print("\nCurrent settings for channel {}".format(self.channel))
        print("==============================")
        for key, (val, unit) in zip(settings.items()):
            print("{:>{num_char}s} {} {}".format(key, val, unit, num_char=longest_key))

    def set_output(self, state):
        """Enables or diables the output of the channel

        Parameters
        ----------
        state : int or str
            "ON" or int 1 to enable
            "OFF" or int 0 to disable

        Raises
        ------
        NotSetError
            If `self.func_gen.verify_param_set` is `True` and the value after
            applying the set function does not match the value returned by the
            get function
        """
        self.func_gen.write("OUTPut{}:STATe {}".format(self.channel, state),
                            custom_err_message="turn channel {} to state {}".format(self.channel, state))
        if self.func_gen.verify_param_set:
            actual_state = self.get_output()
            if not actual_state == state:
                raise NotSetError("Channel {} was not turned {}, it is {}.\nError from the instrument: {}".format(self.channel, state, self.state_str[actual_state], self.func_gen.get_error()))

    def set_function(self, shape):
        """Set the function shape of the output

        Parameters
        ----------
        shape : {SINusoid, SQUare, PULSe, RAMP, PRNoise, <Built_in>, USER[0], USER1, ..., USER255, EMEMory, EFILe}
            <Built_in>::={StairDown|StairUp|Stair Up&Dwn|Trapezoid|RoundHalf|AbsSine|AbsHalfSine|ClippedSine|ChoppedSine|NegRamp|OscRise|OscDecay|CodedPulse|PosPulse|NegPulse|ExpRise|ExpDecay|Sinc|Tan|Cotan|SquareRoot|X^2|HaverSine|Lorentz|Ln(x)|X^3|CauchyDistr|BesselJ|BesselY|ErrorFunc|Airy|Rectangle|Gauss|Hamming|Hanning|Bartlett|Blackman|Laylight|Triangle|DC|Heart|Round|Chirp|Rhombus|Cardiac}

        Raises
        ------
        NotSetError
            If `self.func_gen.verify_param_set` is `True` and the value after
            applying the set function does not match the value returned by the
            get function
        """
        self.func_gen.write("{}FUNCtion:SHAPe {}".format(self.source, shape),
                            custom_err_message="set function {}".format(shape))
        if self.func_gen.verify_param_set:
            actual_shape = self.get_function()
            if not actual_shape == shape:
                raise NotSetError("Function {} was not set on channel {}, it is {}. Check that the function name is correctly spelt. Run set_function.__doc__ to see available shapes.\nError from the instrument: {}".format(shape, self.channel, actual_shape, self.func_gen.get_error()))

    def set_amplitude(self, amplitude):
        """Set the peak-to-peak amplitude in volts

        Parameters
        ----------
        amplitude : float or {"max", "min"}
            0.1mV or four digits resolution, "max" or "min" will set the amplitude
            to the maximum or minimum limit given in `channel_limits`

        Raises
        ------
        NotSetError
            If `self.func_gen.verify_param_set` is `True` and the value after
            applying the set function does not match the value returned by the
            get function
        """
        # check if keyword min or max is given
        if str(amplitude).lower() in ["min", "max"]:
            unit = "" # no unit for MIN/MAX
            # look up what the limit is for this keyword
            amplitude = self.channel_limits["amplitude lims"][0][self.impedance][str(amplitude).lower()]
        else:
            unit = "Vpp"
            # check if the given amplitude is within the current limits
            min_ampl, max_ampl = self.get_amplitude_lims()
            if amplitude < min_ampl or amplitude > max_ampl:
                raise NotSetError("Could not set the amplitude {}{unit} as it is not within the amplitude limits set for the instrument [{}, {}]{unit}".format(amplitude, min_ampl, max_ampl, unit=unit))
        # check that the new amplitude will not violate voltage limits
        min_volt, max_volt = self.get_voltage_lims()
        current_offset = self.get_offset()
        if amplitude/2-current_offset < min_volt or amplitude/2+current_offset > max_volt:
            raise NotSetError("Could not set the amplitude {}{unit} as the amplitude combined with the offset ({}V) will be outside the absolute voltage limits [{}, {}]{unit}".format(amplitude, current_offset, min_volt, max_volt, unit=unit))
        # set the amplitude
        self.func_gen.write("{}VOLTage:LEVel {}{}".format(self.source, amplitude, unit),
                            custom_err_message="set amplitude {}{}".format(amplitude, unit))
        # verify that the amplitude has been set
        if self.func_gen.verify_param_set:
            actual_amplitude = self.get_amplitude()
            # multiply with the appropriate factor according to SI prefix, or
            # if string is empty, use the value looked up from channel_limits earlier
            check_amplitude = amplitude*SI_prefix_to_factor(unit) if not unit == "" else amplitude
            if not actual_amplitude == check_amplitude:
                raise NotSetError("Amplitude {}{} was not set on channel {}, it is {}Vpp. Check that the number is within the possible range and in the correct format.\nError from the instrument: {}".format(amplitude, unit, self.channel, actual_amplitude, self.func_gen.get_error()))

    def set_offset(self, offset, unit="V"):
        """Set offset in volts (or mV, see options)

        Parameters
        ----------
        offset : float
            Unknown resolution, guessing 0.1mV or four digits resolution
        unit : {mV, V}, default V

        Raises
        ------
        NotSetError
            If `self.func_gen.verify_param_set` is `True` and the value after
            applying the set function does not match the value returned by the
            get function
        """
        # check that the new offset will not violate voltage limits
        min_volt, max_volt = self.get_voltage_lims()
        current_amplitude = self.get_amplitude()
        if current_amplitude/2-offset < min_volt or current_amplitude/2+offset > max_volt:
            raise NotSetError("Could not set the offset {}{unit} as the offset combined with the amplitude ({}V) will be outside the absolute voltage limits [{}, {}]{unit}".format(offset, current_amplitude, min_volt, max_volt, unit=unit))
        # set the offset
        self.func_gen.write("{}VOLTage:LEVel:OFFSet {}{}".format(self.source, offset, unit),
                            custom_err_message="set offset {}{}".format(offset, unit))
        # verify that the offset has been set
        if self.func_gen.verify_param_set:
            actual_offset = self.get_offset()
            # multiply with the appropriate factor according to SI prefix
            check_offset = offset*SI_prefix_to_factor(unit)
            if not actual_offset == check_offset:
                raise NotSetError("Offset {}{} was not set on channel {}, it is {}V. Check that the number is within the possible range and in the correct format.\nError from the instrument: {}".format(offset, unit, self.channel, actual_offset, self.func_gen.get_error()))

    def set_frequency(self, freq, unit="Hz"):
        """Set the frequency in Hertz (or kHz, MHz, see options)

        Parameters
        ----------
        freq : float
            The resolution is 1 μHz or 12 digits.
        unit : {Hz, kHz, MHz}, default Hz

        Raises
        ------
        NotSetError
            If `self.func_gen.verify_param_set` is `True` and the value after
            applying the set function does not match the value returned by the
            get function
        """
        if str(freq).lower() in ["min", "max"]: # handle min and max keywords
            unit = "" # no unit for MIN/MAX
            # look up what the limit is for this keyword
            freq = self.channel_limits["frequency lims"][0][str(freq).lower()]
        else:
            # check if the given frequency is within the current limits
            min_freq, max_freq = self.get_frequency_lims()
            if freq < min_freq or freq > max_freq:
                raise NotSetError("Could not set the frequency {}{} as it is not within the frequency limits set for the instrument [{}, {}]Hz".format(freq, unit, min_freq, max_freq))
        # check that the new amplitude will not violate voltage limits
        min_volt, max_volt = self.get_voltage_lims()
        # set the frequency
        self.func_gen.write("{}FREQuency:FIXed {}{}".format(self.source, freq, unit),
                            custom_err_message="set frequency {}{}".format(freq, unit))
        # verify that the amplitude has been set
        if self.func_gen.verify_param_set:
            actual_freq = self.get_frequency()
            # multiply with the appropriate factor according to SI prefix, or
            # if string is empty, use the value looked up from channel_limits earlier
            check_freq = freq*SI_prefix_to_factor(unit) if not unit == "" else freq
            if not actual_freq == check_freq:
                raise NotSetError("Frequency {}{} was not set on channel {}, it is {}Hz. Check that the number is within the possible range and in the correct format.\nError from the instrument: {}".format(freq, unit, self.channel, actual_freq, self.func_gen.get_error()))


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AUXILIARY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

class NotSetError(Exception):
    """Error for when a value cannot be written to the instrument"""
    pass

class NotCompatibleError(Exception):
    """Error for when the instrument is not compatible with this module"""
    pass

def SI_prefix_to_factor(unit):
    """Convert an SI prefix to a numerical factor

    Parameters
    ----------
    unit : str
        The unit whose first character is checked against the list of prefactors
        {"M": 1e6, "k": 1e3, "m": 1e-3}

    Returns
    -------
    factor : float or `None`
        The appropriate factor or 1 if not found in the list, or `None` if the unit string is empty
    """
    # SI prefix to numerical value
    SI_conversion = {"M": 1e6, "k": 1e3, "m": 1e-3}
    try: # using the unit's first character as key in the dictionary
        factor = SI_conversion[unit[0]]
    except KeyError: # if the entry does not exist
        factor = 1
    except IndexError: # if the unit string is empty
        factor = None
    return factor


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EXAMPLES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def example_print_current_settings(address):
    """Example showing how to connect and get the current settings of the instrument"""
    with func_gen(address) as fgen:
        fgen.print_settings()

def example_lock_frequencies(address):
    """Example showing the frequency being set to 10Hz and then the frequency
    lock enabled, using the frequency at ch1 as the common frequency"""
    with func_gen(address, verbose=False) as fgen:
        fgen.ch1.set_frequency(10)
        fgen.enable_frequency_lock(use_channel=1)

def example_set_and_use_custom_waveform(fgen=None, channel=1, plot_signal=True):
    """Example showing a waveform being created, transferred to the instrument,
    and applied to a channel"""
    # create a signal
    x = np.linspace(0, 4*np.pi, 8000)
    signal = np.sin(x)+x/5
    if plot_signal: # plot the signal for visual control
        import matplotlib.pyplot as plt
        plt.plot(signal)
        plt.show()
    # create initialise fgen if it was not supplied
    if fgen is None:
        fgen = func_gen()
        close_fgen = True # specify that it should be closed at end of function
    else:
        close_fgen = False # do not close the supplied fgen at end
    print("Current waveform catalogue")
    for i, wav in enumerate(fgen.get_waveform_catalogue()): print("  {}: {}".format(i, wav))
    # transfer the waveform
    fgen.set_custom_waveform(signal, memory_num=5, verify=True)
    print("New waveform catalogue:")
    for i, wav in enumerate(fgen.get_waveform_catalogue()): print("  {}: {}".format(i, wav))
    print("Set new wavefrom to channel {}..".format(channel), end=" ")
    fgen.channels[channel-1].set_output("OFF")
    fgen.channels[channel-1].set_function("USER5")
    print("ok")
    # print current settings
    fgen.get_settings()
    if close_fgen: fgen.close()


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN FUNCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

if __name__ == '__main__':
    address = 'USB0::0x0699::0x0353::1731975::INSTR'
    example_print_current_settings(address)
    example_lock_frequencies(address)
    with func_gen(address) as fgen:
        example_set_and_use_custom_waveform(fgen)
