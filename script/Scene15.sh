#!/bin/bash
cleanup() {
  echo "检测到中断信号，正在终止所有后台任务..."
  kill 0
}

trap cleanup SIGINT SIGTERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 定义统一的日志目录变量，防止手误写错
LOG_DIR="test-Mixture_Scene15"
MODEL="Ours"
output_dir="${LOG_DIR}"
# 确保输出目录存在
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
  echo "Created log directory: $LOG_DIR"
fi

echo "启动任务]"

CUDA_VISIBLE_DEVICES=0 python main_train.py \
    --config_file config/Scene15.yaml \
    --train_type Mixture \
    --test_type Mixture \
    --which_model Ours \
    --fp_ratio 0 \
    --missing_rate 0 \
    --incomplete_ratio 0.1 \
    --missing_loss_weight 0.2 \
    --sparse_topk 3 \
    --sparse_temp 0.5 \
    --missing_start_epoch 150 \
    --min_missing_conf 0.9 \
    --shuffle_noisy_view view2 \
    --realign_anchor_view view1 \
    --output_dir "${output_dir}" \
    >> "${LOG_DIR}/${MODEL}_train_fp0_mr0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=0 python main_train.py \
    --config_file config/Scene15.yaml \
    --train_type Mixture \
    --test_type Mixture \
    --which_model Ours \
    --fp_ratio 0.8 \
    --missing_rate 0.8 \
    --incomplete_ratio 0.1 \
    --missing_loss_weight 0.2 \
    --sparse_topk 3 \
    --sparse_temp 0.5 \
    --missing_start_epoch 150 \
    --min_missing_conf 0.9 \
    --shuffle_noisy_view view2 \
    --realign_anchor_view view1 \
    --output_dir "${output_dir}" \
    >> "${LOG_DIR}/${MODEL}_train_fp0p8_mr0p8.log" 2>&1 &

CUDA_VISIBLE_DEVICES=0 python main_train.py \
    --config_file config/Scene15.yaml \
    --train_type Mixture \
    --test_type Mixture \
    --which_model Ours \
    --fp_ratio 0.2 \
    --missing_rate 0.2 \
    --incomplete_ratio 0.1 \
    --missing_loss_weight 0.2 \
    --sparse_topk 3 \
    --sparse_temp 0.5 \
    --missing_start_epoch 150 \
    --min_missing_conf 0.9 \
    --shuffle_noisy_view view2 \
    --realign_anchor_view view1 \
    --output_dir "${output_dir}" \
    >> "${LOG_DIR}/${MODEL}_train_fp0p2_mr0p2.log" 2>&1 &

CUDA_VISIBLE_DEVICES=0 python main_train.py \
    --config_file config/Scene15.yaml \
    --train_type Mixture \
    --test_type Mixture \
    --which_model Ours \
    --fp_ratio 0.5 \
    --missing_rate 0.5 \
    --incomplete_ratio 0.1 \
    --missing_loss_weight 0.2 \
    --sparse_topk 3 \
    --sparse_temp 0.5 \
    --missing_start_epoch 150 \
    --min_missing_conf 0.9 \
    --shuffle_noisy_view view2 \
    --realign_anchor_view view1 \
    --output_dir "${output_dir}" \
    >> "${LOG_DIR}/${MODEL}_train_fp0p5_mr0p5.log" 2>&1 &
wait
echo "所有实验完成！日志请查看 $LOG_DIR 文件夹。"