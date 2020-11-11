v0.4.0
- Now compatible with `pyvisa v11.1` as in this version the `write()` will not return the status code anymore
- Added `check_pyvisa_status()`, now checking status for both queries and writes

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
