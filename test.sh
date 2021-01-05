#!/bin/bash
PROJECT="/home/jaliaga/bioPDSN/"
cd "$PROJECT"
function runTest()
{
    python test.py --test_database "hard" --model ${1} -b 64 -num_workers 8
    python test.py --test_database "mask" --model ${1} -b 64 -num_workers 8
    python test.py --test_database "nonmask" --model ${1} -b 64 -num_workers 8
}
runTest BIOAPI