v0.5.0
- Adding `_` suggesting private methods and functions:
  - `FuncGen` methods
    - `_check_pyvisa_status()`
    - `_initialise_model_properties()`
    - `_normalise_to_waveform()`
    - `_verify_waveform()`
    - `_check_arb_waveform_length()`
    - `_check_arb_waveform_type_and_range()`
    - `_impedance_dependent_limit()`
    - `_spawn_channel()`
- `FuncGen` attributes
    - `_id`
    - `_inst`
    - `address` -> `_visa_address`
    - `_arbitrary_waveform_length`
    - `_arbitrary_waveform_resolution`
    - `_override_compatibility`
    - `_timeout`
    - `_maker`
    - `_serial`
    - `_model`
  - `FuncGenChannel`
    - `_source`
    - `state_str` -> `_state_to_str`


v0.4.0
- Ensuring compatibility with `pyvisa v11.1`: ([issue #2](https://github.com/asvela/tektronix-func-gen/issues/2))
  - PYVISAs `write()` does not return the status code anymore, so the module is modified accordingly
  - not necessary to have an empty query after `write_binary_values()` in `set_custom_waveform()`
- Added `check_pyvisa_status()`, now checking status for both queries and writes
- Bug fixes for `set_frequency()` and `set_offset()` that were previously not taking into account the unit when calculating if it was within the limits

v0.3.1
- Added note about known issue with TekVISA
- Added more details about flat waveform offset through custom waveform workaround

v0.3.0
- Made SI_prefix_to_factor a staticmethod in FuncGen rather than module-level function
- Added note about constant/flat function

v0.2
- More PEP8 compliant:
  - `func_gen()` -> `FuncGen()`
  - `func_gen_channel()` -> `FuncGenChannel()`
  - All lines < 100 characters (mostly < 80)
- No more `enable`/`disable_frequency_lock()`, now `set_frequency_lock()`
- Settings dictionary now contains tuples of value and unit, e.g. `settings = {"amplitude": (3, "Vpp"), ..}`
- Implemented `set_settings()` in both `FuncGen` and `FuncGenChannel` that takes a settings dictionary as input
- `get`/`set_output()` links to `get`/`set_output_state()`
- More examples
- Expanded README

v0.1
- First release
