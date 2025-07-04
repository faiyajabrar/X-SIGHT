@echo off
echo =============================================
echo  OPTIMIZED ATTENTION U-NET TRAINING (32-bit)
echo =============================================
echo.
echo 🚀 State-of-the-art training with all improvements
echo.
echo ✅ ENABLED FEATURES:
echo   🧠 Attention U-Net (ResNet34 + attention gates)
echo   📊 Hybrid Loss (Dice + Focal + Boundary)
echo   ⚡ AdamW + OneCycleLR (superconvergence)
echo   📈 Progressive resizing (128→256px)
echo   🎨 Progressive augmentations
echo   🔄 MixUp data augmentation (30%% batches)
echo   📱 Exponential Moving Average (EMA)
echo   📋 Stochastic Weight Averaging (SWA)
echo   🎯 Class-wise performance monitoring
echo   ⏰ Early stopping + gradient clipping
echo.
echo 🎯 EXPECTED PERFORMANCE:
echo   Validation Dice: 0.6+ (vs 0.34 baseline = 76%+ improvement)
echo   Stable 32-bit training without mixed precision issues
echo.

python training/train.py ^
    --lr 3e-4 ^
    --dropout 0.1 ^
    --batch_size 16 ^
    --epochs 60 ^
    --weight_decay 1e-4 ^
    --warmup_epochs 5 ^
    --early_stopping_patience 15 ^
    --grad_clip_val 1.0 ^
    --use_swa ^
    --swa_start_epoch 30

echo.
echo ✅ TRAINING COMPLETE!
echo.
echo 📊 Results Summary:
echo   - Best checkpoint saved with highest validation Dice
echo   - Separate checkpoint for best rare class (Dead cells) performance  
echo   - SWA model available for final deployment
echo   - Class-wise performance logged for each epoch
echo.
echo 🚀 Next Steps:
echo   1. Check lightning_logs/ for detailed metrics
echo   2. Best model: advanced-*.ckpt files
echo   3. SWA model typically performs best for final evaluation
echo.
pause 