# Tektronix arbitrary function generator control through PyVISA

v0.1.0 // Dec 2019

API documentation available [in docs/tektronix_func_gen.html](docs/tektronix_func_gen.html). (To build the documentation use [pdoc3](https://pdoc3.github.io/pdoc/) and run `$ pdoc --html tektronix_func_gen`.)


## Installation

Put the module file in the folder wherein the file you will import it from resides.


## Usage

An example:

```python
import tektronix_func_gen as tfg

with tfg.func_gen('VISA ADDRESS OF YOUR INSTRUMENT') as fgen:
      fgen.ch1.set_function("SIN")
      fgen.ch1.set_frequency(25, unit="Hz")
      fgen.ch1.set_output("ON")
      fgen.ch2.set_output("OFF")
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
   offset 0.0   -0.45 V
frequency 25.0  10.0  Hz
```

More examples are included at the end of the module.

### Impedance

Unfortunately the impedance (50 or high Z) cannot be controlled or read remotely. Which setting is in use affects the limits of the output voltage. Use the optional impedance keyword in the initialisation of the func_gen object to make the object aware what limits applies: `func_gen('VISA ADDRESS OF YOUR INSTRUMENT', impedance=("highZ", "50ohm"))`.


### Syncronisation and frequency lock

The phase of the two channels can be syncronised with `syncronise_waveforms()`. Frequency lock can also be enabled/disabled with `enable_frequency_lock()`/`disable_frequency_lock()`.


### Arbitrary waveforms

14 bit vertical resolution arbitrary waveforms can be transferred to the 256 available user defined functions on the function generator.
The length of the waveform must be between 2 and 8192 points.

```python
import numpy as np
import tektronix_func_gen as tfg
with tfg.func_gen('VISA ADDRESS OF YOUR INSTRUMENT') as fgen:
      x = np.linspace(0, 4*np.pi, 8000)
      waveform = np.sin(x)+x/5
      print("Current waveform catalogue")
      for i, wav in enumerate(fgen.get_waveform_catalogue()): print("  {}: {}".format(i, wav))
      # transfer the waveform
      fgen.set_custom_waveform(waveform, memory_num=5, verify=True)
      print("New waveform catalogue:")
      for i, wav in enumerate(fgen.get_waveform_catalogue()): print("  {}: {}".format(i, wav))
      print("Set new wavefrom to channel {}..".format(channel), end=" ")
      fgen.channels[channel-1].set_output("OFF")
      fgen.channels[channel-1].set_function("USER5")
      print("ok")
      # print current settings
      fgen.get_settings()
```


### Set voltage and frequency limits

Limits for amplitude, voltage and frequency can be set by accessing the `func_gen_channel.channel_limits` dictionary or using the `set_stricter_limits()`. The dictionary has the following structure (these are the standard limits for AFG1022)

```python
channel_limits = {
  "frequency lims": ({"min": 1e-6, "max": 25e6}, "Hz"),
  "voltage lims":   ({"50ohm": {"min": -5, "max": 5},
                      "highZ": {"min": -10, "max": 10}}, "V"),
  "amplitude lims": ({"50ohm": {"min": 0.001, "max": 10},
                      "highZ": {"min": 0.002, "max": 20}}, "Vpp")}
```
