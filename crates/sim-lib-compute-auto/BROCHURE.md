# Auto Compute

`sim-lib-compute-auto` provides `site/compute/auto`, the stable automatic tensor
placement surface. Today it selects the modeled provider when configured and
falls back to CPU when no compatible profile is present.

