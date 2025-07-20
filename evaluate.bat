@echo off
echo 🚀 Running Attention U-Net Model Evaluation...
echo.

REM Run the evaluation script with the best checkpoint
python evaluate_results.py --checkpoint "lightning_logs/version_1/checkpoints/advanced-epoch=112-val_dice=0.656.ckpt" --num_samples 6 --save_dir "evaluation_results"

echo.
echo ✅ Evaluation complete! Check evaluation_results folder for saved visualizations.
pause 