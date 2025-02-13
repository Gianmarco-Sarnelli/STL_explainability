#!/bin/bash

test_names=("nountill_M2M" "nountill_M2E" "nountill_M2B" "nountill_M2G" "nountill_M2S" "nountill_E2M" "nountill_E2E" "nountill_E2B" "nountill_E2G" "nountill_E2S" "nountill_B2M" "nountill_B2E" "nountill_B2B" "nountill_B2G" "nountill_B2S" "nountill_G2M" "nountill_G2E" "nountill_G2B" "nountill_G2G" "nountill_G2S" "nountill_S2M" "nountill_S2E" "nountill_S2B" "nountill_S2G" "nountill_S2S")


for testname in "${test_names[@]}"; do
    cat output_${testname}_{0..17}.log > Outputs/output_${testname}.log
    cat error_${testname}_{0..17}.log > Errors/error_${testname}.log
done
