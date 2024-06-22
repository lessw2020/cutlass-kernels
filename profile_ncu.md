/opt/nvidia/nsight-compute/2023.2.1/ncu --target-processes all --set full 
--import-source yes -f --section SchedulerStats --section WarpStateStats --section SpeedOfLight_RooflineChart 
--section SpeedOfLight_HierarchicalTensorRooflineChart --section MemoryWorkloadAnalysis_Chart 
-o vllm_scaled_mm python vllm_scaled_mm.py
