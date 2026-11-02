E4 complete result artifact archive (tensor checkpoints excluded)

All non-safetensors files in run/final_artifact_manifest.json are included.
This includes all original eval/train/tune trajectories, point/peek records,
replay outcomes, judge requests and scores, preference pairs, mechanism outputs,
adapter configs and training logs. No scientific file is changed.
The separately published e4.json, E4_REPORT.txt and verification reports contain
the complete registered findings. LoRA tensor checkpoints remain on the server;
their exact paths, sizes and SHA256 hashes are listed in archive_manifest.json.

From exploratory_E4/results_archive/:
  cat e4_nonweight_results.tar.gz.part* > /tmp/e4_nonweight_results.tar.gz
  mkdir -p /tmp/e4_results
  tar -xzf /tmp/e4_nonweight_results.tar.gz -C /tmp/e4_results
The extracted data/ tree recreates the archived non-weight experiment outputs.
Check each part SHA256 against archive_manifest.json before extraction; every
member was decompressed and checked against the original final manifest before
publication. Checkpoints are not needed to reproduce reported statistics from
the recorded outcomes, but are needed to run the trained policies again.
