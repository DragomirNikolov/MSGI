# MSGI
MSGI (Model satellite glaciation intercomparisson) is a project comparing cloud glaciation observed from satellites to those occuring in climate models.

Model preprocessing accepts an optional `mode` in the MSGI YAML config:

```yaml
mode: cirrus
```

- `mixed_phase` (default when omitted): six 6-degree temperature bins spanning −36 to 0.
- `cirrus`: six 10-degree temperature bins with edges −98, −88, −78, −68, −58, −48, and −38.

Both modes use the existing 273 K temperature offset. Cirrus therefore filters
175–235 K, including each bin's lower boundary and excluding its upper boundary.
Bins are processed from warmest to coldest; cirrus output filenames use temperature
labels `48_38`, `58_48`, …, `98_88`. The selected range also filters the combined
additional-variable output, with the existing total-water-content threshold.

See [the example config](msgi/aux/msgi_config_example.yaml). Use separate
`preprocessing_output_dir` values to retain results from both modes, since the
combined output filenames are shared.
