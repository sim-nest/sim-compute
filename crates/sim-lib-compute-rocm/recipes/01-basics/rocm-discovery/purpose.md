# ROCm discovery

Shows that ROCm is an optional compute site: absent runtime libraries or missing
AMD `gfx*` evidence produce no exported site, while a validated runtime can
advertise dense matmul.
