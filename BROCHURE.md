# SIM Compute

Portable tensor execution sites for SIM.

`sim-compute` lets a tensor expression run at a modeled compute site without
changing the canonical tensor value or the kernel. It is useful for testing
placement, resident allocation lifetimes, fault handling, and automatic CPU
fallback before hardware providers are enabled.

