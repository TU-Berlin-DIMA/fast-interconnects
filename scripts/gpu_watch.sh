#!/usr/bin/env sh

watch -n1 --differences=cumulative nvidia-smi --query-gpu="index,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory" --format=csv
