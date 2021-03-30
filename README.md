## Tektronix arbitrary function generator control

[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/asvela/tektronix-func-gen?style=flat-square)](https://www.codefactor.io/repository/github/asvela/tektronix-func-gen)
[![MIT License](https://img.shields.io/github/license/asvela/dlc-control?style=flat-square)](https://github.com/asvela/dlc-control/blob/main/LICENSE)

Provides basic control of AFG1000 and AFG3000 series Tektronix Arbitrary Function 
Generators, possibly also others. This includes setting basic settings such as
selecting functions, transferring or selecting custom waveforms, amplitude and offset
control, phase syncronisation and frequency locking.

API documentation available [here](https://asvela.github.io/tektronix-func-gen/),
or in the repository [docs/index.html](docs/index.html). (To build the documentation
yourself use [pdoc3](https://pdoc3.github.io/pdoc/) and run
`$ python3 pdoc --html -o ./docs/ tektronix_func_gen`.)

Tested on Win10 with NI-VISA and PyVISA v1.11 (if using PyVISA <v1.11 use <v0.4
of this module).


### Known issues

- **For TekVISA users:** a `pyvisa.errors.VI_ERROR_IO` is raised unless the
  Call Monitor application that comes with TekVISA is open and capturing
  (see issue [#1](https://github.com/asvela/tektronix-func-gen/issues/1)).
  NI-VISA does not have this issue.
- The offset of the built-in DC (flat) function cannot be controlled directly. A
  workaround is to transfer a flat custom waveform to a memory location,
  see [Flat function offset control](#flat-function-offset-control) in this readme.
- The frequency limits can in practice be stricter than what is set by the module,
  as the module is using the limits for a sine, where as other functions, such as
  ramp might have lower limit

### Installation

Put the module file in the folder wherein the Python file you will import it
from resides.

**Dependencies:**

  - The package needs VISA to be installed. It is tested with NI-VISA,
    *TekVISA might not work*, see `Known issues`
  - The Python packages `numpy` and `pyvisa` (>=v1.11) are required


### Usage (through examples)

An example of basic control

```python
import tektronix_func_gen as tfg

with tfg.FuncGen('VISA ADDRESS OF YOUR INSTRUMENT') as fgen:
      fgen.ch1.set_function("SIN")
      fgen.ch1.set_frequency(25, unit="Hz")
      fgen.ch1.set_offset(50, unit="mV")
      fgen.ch1.set_amplitude(0.002)
      fgen.ch1.set_output("ON")
      fgen.ch2.set_output("OFF")
      # alternatively fgen.ch1.print_settings() to show from one channel only
      fgen.print_settings()
```

yields something like (depending on the settings already in use)

```
Connected to TEKTRONIX model AFG1022, serial XXXXX

Current settings for TEKTRONIX AFG1022 XXXXX

  Setting Ch1   Ch2   Unit
==========================
   output ON    OFF    
 function SIN   RAMP  
amplitude 0.002 1     Vpp
   offset 0.05  -0.45 V
frequency 25.0  10.0  Hz
```

Settings can also be stored and restored:

```python
"""Example showing how to connect, get the current settings of
the instrument, store them, change a setting and then restore the
initial settings"""
import tektronix_func_gen as tfg
with tfg.FuncGen('VISA ADDRESS OF YOUR INSTRUMENT') as fgen:
    fgen.print_settings()
    print("Saving these settings..")
    settings = fgen.get_settings()
    print("Change to 1Vpp amplitude for channel 1..")
    fgen.ch1.set_amplitude(1)
    fgen.print_settings()
    print("Reset back to initial settings..")
    fgen.set_settings(settings)
    fgen.print_settings()
```


#### Syncronisation and frequency lock

The phase of the two channels can be syncronised with `syncronise_waveforms()`.
Frequency lock can also be enabled/disabled with `set_frequency_lock()`:

```python
"""Example showing the frequency being set to 10Hz and then the frequency
lock enabled, using the frequency at ch1 as the common frequency"""
import tektronix_func_gen as tfg
with tfg.FuncGen('VISA ADDRESS OF YOUR INSTRUMENT', verbose=False) as fgen:
    fgen.ch1.set_frequency(10)
    fgen.set_frequency_lock("ON", use_channel=1)
```


#### Arbitrary waveforms

14 bit vertical resolution arbitrary waveforms can be transferred to the 256
available user defined functions on the function generator.
The length of the waveform must be between 2 and 8192 points.

```python
import numpy as np
import tektronix_func_gen as tfg
with tfg.FuncGen('VISA ADDRESS OF YOUR INSTRUMENT') as fgen:
      # create waveform
      x = np.linspace(0, 4*np.pi, 8000)
      waveform = np.sin(x)+x/5
      # transfer the waveform (normalises to the vertical waveform range)
      fgen.set_custom_waveform(waveform, memory_num=5, verify=True)
      # done, but let's have a look at the waveform catalogue ..
      print("New waveform catalogue:")
      for i, wav in enumerate(fgen.get_waveform_catalogue()): print("  {}: {}".format(i, wav))
      # .. and set the waveform to channel 1
      print("Set new wavefrom to channel 1..", end=" ")
      fgen.ch1.set_output("OFF")
      fgen.ch1.set_function("USER5")
      print("ok")
      # print current settings
      fgen.print_settings()
```

##### Flat function offset control

The offset of the built-in DC function cannot be controlled (the offset command
simply does not work, an issue from Tektronix). A workaround is to transfer a
flat custom waveform (two or more points of half the vertical range
(`arbitrary_waveform_resolution`)) to a memory location:

```python
with tfg.FuncGen('VISA ADDRESS OF YOUR INSTRUMENT') as fgen:
    flat_wfm = int(fgen.arbitrary_waveform_resolution/2)*np.ones(2).astype(np.int32)
    fgen.set_custom_waveform(flat_wfm, memory_num=255, normalise=False)
    fgen.ch1.set_function("USER255")
    fgen.ch1.set_offset(2)
```

Note the `normalise=False` argument.


#### Set voltage and frequency limits

Limits for amplitude, voltage and frequency for each channel are kept in a
dictionary `FuncGenChannel.channel_limits` (these are the standard limits
  for AFG1022)

```python
channel_limits = {
  "frequency lims": ({"min": 1e-6, "max": 25e6}, "Hz"),
  "voltage lims":   ({"50ohm": {"min": -5, "max": 5},
                      "highZ": {"min": -10, "max": 10}}, "V"),
  "amplitude lims": ({"50ohm": {"min": 0.001, "max": 10},
                      "highZ": {"min": 0.002, "max": 20}}, "Vpp")}
```

They chan be changed by `FuncGenChannel.set_limit()`, or by using the
`FuncGenChannel.set_stricter_limits()` for a series of prompts.

```python
import tektronix_func_gen as tfg
"""Example showing how limits can be read and changed"""
with tfg.FuncGen('VISA ADDRESS OF YOUR INSTRUMENT') as fgen:
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
```


#### Impedance

Unfortunately the impedance (50Ω or high Z) cannot be controlled or read remotely.
Which setting is in use affects the limits of the output voltage. Use the optional
impedance keyword in the initialisation of the `FuncGen` object to make the object
aware what limits applies: `FuncGen('VISA ADDRESS OF YOUR INSTRUMENT', impedance=("highZ", "50ohm"))`.
