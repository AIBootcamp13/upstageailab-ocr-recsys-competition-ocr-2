#!/bin/bash
echo "=== SYSTEM RESOURCE MONITOR ==="
echo "Timestamp: $(date)"
echo

echo "=== TOP CPU PROCESSES ==="
ps aux --sort=-%cpu | head -10
echo

echo "=== TOP MEMORY PROCESSES ==="
ps aux --sort=-%mem | head -10
echo

echo "=== ORPHANED PROCESSES ==="
ps -eo pid,ppid,stat,cmd | grep -v "PPID" | while read pid ppid stat cmd; do
    if [ "$ppid" != "1" ] && [ "$stat" != "Z" ]; then
        if ! ps -p $ppid > /dev/null 2>&1; then
            echo "ORPHANED: PID $pid PPID $ppid CMD: $cmd"
        fi
    fi
done
echo

echo "=== ZOMBIE PROCESSES ==="
ps aux | awk '{if ($8 == "Z") print "ZOMBIE:", $0}'
echo

echo "=== LONG RUNNING PROCESSES (>1 hour) ==="
ps -eo pid,etime,cmd | grep -v "00:" | grep -E "([0-9]{2}:| [0-9]+-)" | head -10
echo

echo "=== DISK USAGE ==="
df -h | head -5
echo

echo "=== MEMORY INFO ==="
free -h
