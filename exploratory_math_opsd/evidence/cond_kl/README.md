# Reproduce the conditional KL experiment

Use the retained OPSD conda environment, four A100s, archived training source, model and probe data. Paths and training settings follow the registered R run. Install the retained environment using the mount section of exploratory_math_opsd/run_four.sh. In this directory:

1. Run `python test_objective.py`.
2. Run `CUDA_VISIBLE_DEVICES=i python preflight.py i` for i=0,1,2,3, then `python preflight.py report` and `python report.py preflight`.
3. Run `python train.py R-noclip` then `python train.py R-cond`. Training creates a separate copy of the deployed source under the retained output directory. The frozen teacher, optimizer schedule, generation, four ranks and gradient accumulation use the R entry point.
4. Run `bash run_measurements.sh` to aggregate actual training timings and run the fixed probe, own-prefix probe and benchmark cells. Each stage creates a separate commit. Set COND_PUBLISH_SCRIPT to the environment's path-scoped publisher to upload each stage. All primary answer data and checkpoint hashes remain on the mounted disk.
