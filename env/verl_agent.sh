# Isolated native verl-agent environment; source from bash.
FORKAUDIT_VERL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$FORKAUDIT_VERL_ROOT/.runtime/venvs/verl-agent-20bd331b/bin:$PATH"
export VERL_AGENT_SOURCE="$FORKAUDIT_VERL_ROOT/.runtime/external/verl-agent-20bd331b"
export M0_MODEL="$FORKAUDIT_VERL_ROOT/.runtime/models/Qwen2.5-7B-Instruct"
export ALFWORLD_DATA="$FORKAUDIT_VERL_ROOT/.runtime/verl-agent-assets"
export HF_HOME="$FORKAUDIT_VERL_ROOT/.runtime/cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export RAY_TMPDIR=/tmp/forkaudit_verl
unset VLLM_BATCH_INVARIANT TRITON_F32_DEFAULT NVIDIA_TF32_OVERRIDE CUBLAS_WORKSPACE_CONFIG
