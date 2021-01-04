#!/bin/bash
PROJECT="/home/jaliaga/bioPDSN/"
cd "$PROJECT"
function runTest()
{
    python test.py --test_database ${1} --model ${2} -b 64 -num_workers 8
}
runTest hard CASIA-TRIPLET
runTest mask CASIA-TRIPLET
