#!/bin/bash
# Monitor experiment progress
while true; do
    echo "=== $(date) ==="
    tail -5 /mnt/e/Project/SSM-World-Model/experiments/run_10seeds.log 2>/dev/null
    echo "---"
    ps aux | grep run_10seeds | grep -v grep | wc -l
    echo "processes running"
    sleep 60
done
