#!/usr/bin/env bash

cloc --include-lang=Rust,C++,CUDA,"C/C++ Header" --exclude-dir=target --git --diff master wide-tuples --out=changes.log
convert -size 4000x4000 xc:white -font "FreeMono" -pointsize 64 -fill black -annotate +15+80 "@changes.log" -trim changes.png
