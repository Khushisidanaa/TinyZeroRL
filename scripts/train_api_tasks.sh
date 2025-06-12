#!/bin/bash

# API Tasks RL Training Script
echo "🚀 Starting API Tasks RL Training..."

# Validate required environment variables
if [ -z "$BASE_MODEL" ]; then
    echo "❌ Error: BASE_MODEL environment variable not set"
    echo "Please set: export BASE_MODEL=/path/to/your/model"
    exit 1
fi

if [ -z "$DATA_DIR" ]; then
    echo "❌ Error: DATA_DIR environment variable not set"
    echo "Please set: export DATA_DIR=/path/to/your/data"
    exit 1
fi

if [ -z "$N_GPUS" ]; then
    echo "⚠️  Warning: N_GPUS not set, defaulting to 1"
    export N_GPUS=1
fi

if [ -z "$ROLLOUT_TP_SIZE" ]; then
    echo "⚠️  Warning: ROLLOUT_TP_SIZE not set, defaulting to 1"
    export ROLLOUT_TP_SIZE=1
fi

if [ -z "$EXPERIMENT_NAME" ]; then
    export EXPERIMENT_NAME="api-tasks-rl-$(date +%Y%m%d-%H%M%S)"
fi

echo "📋 Training Configuration:"
echo "  Model: $BASE_MODEL"
echo "  Data: $DATA_DIR"
echo "  GPUs: $N_GPUS"
echo "  Rollout TP Size: $ROLLOUT_TP_SIZE"
echo "  Experiment: $EXPERIMENT_NAME"
echo ""

# Run the training with the same parameters as your current setup
python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=64 \
data.val_batch_size=128 \
data.max_prompt_length=256 \
data.max_response_length=1024 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.ppo_micro_batch_size=2 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.grad_offload=False \
actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
actor_rollout_ref.ref.fsdp_config.param_offload=False \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.ppo_micro_batch_size=2 \
critic.model.enable_gradient_checkpointing=True \
critic.model.fsdp_config.param_offload=False \
critic.model.fsdp_config.grad_offload=False \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.critic_warmup=0 \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=10 \
trainer.test_freq=10 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee api_tasks_training.log

echo ""
echo "✅ Training completed! Check api_tasks_training.log for details." 