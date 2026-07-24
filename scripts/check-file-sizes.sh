#!/bin/sh
set -eu

tmp="${TMPDIR:-/tmp}/sim-compute-file-sizes.$$"
: > "$tmp"
find crates xtask -type f -name '*.rs' | while IFS= read -r file; do
  lines=$(wc -l < "$file")
  if [ "$lines" -gt 500 ]; then
    echo "$file exceeds 500 lines ($lines)" >&2
    echo "$file" >> "$tmp"
  fi
done
if [ -s "$tmp" ]; then
  rm -f "$tmp"
  exit 1
fi
rm -f "$tmp"
