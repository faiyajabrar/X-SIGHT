@echo off
echo =============================================
echo  ADVANCED TRAINING - RESUME
echo =============================================
echo.
echo 🔄 Resuming advanced Attention U-Net training
echo.
echo This will automatically:
echo   📁 Find and load the latest checkpoint
echo   🔄 Restore exact training state (epoch, metrics, lr)
echo   📊 Use same data split for consistency
echo   🎯 Continue progressive training
echo   ⚡ Preserve optimizer and scheduler state
echo.
echo ⏱️  Training will continue from last completed epoch
echo 📊 All metrics and progress will be preserved
echo.
echo 📚 Using virtual environment Python...
echo.

venv\Scripts\python.exe training/train.py ^
    --resume ^
    --lr 3e-4 ^
    --batch_size 16 ^
    --epochs 60 ^
    --grad_clip_val 1.0 ^
    --early_stopping_patience 15

echo.
echo ✅ RESUMED TRAINING COMPLETE!
echo.
echo 🎯 Check lightning_logs/ for:
echo   - Model checkpoints (.ckpt files)
echo   - Training state (training_state.json)
echo   - Progress logs and metrics
echo.
echo 💡 Training state saved after each epoch for safe resuming
echo.
pause 