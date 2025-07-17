@echo off
echo =============================================
echo  OPTIMIZED ATTENTION U-NET TRAINING (32-bit)
echo  🏆 TUNED HYPERPARAMETERS (Trial #281)
echo =============================================
echo.
echo 🚀 State-of-the-art training with all improvements
echo 🏆 OPTIMIZED HYPERPARAMETERS:
echo   📚 Learning Rate: 2.11e-4 (vs 3e-4 default)
echo   🎯 Dropout: 0.09 (vs 0.1 default) 
echo   ⚖️  Weight Decay: 1.04e-3 (vs 1e-4 default)
echo   ✂️  Gradient Clipping: 1.83 (vs 1.0 default)
echo.
echo ✅ ENABLED FEATURES:
echo   🧠 Attention U-Net (ResNet34 + attention gates)
echo   📊 Hybrid Loss (Dice + Focal + Boundary)
echo   ⚡ AdamW + OneCycleLR (superconvergence)
echo   📈 Progressive resizing (128→256px, full size at epoch 84)
echo   🎨 Progressive augmentations
echo   🔄 MixUp data augmentation (30%% batches)
echo   📱 Exponential Moving Average (EMA)
echo   📋 Stochastic Weight Averaging (SWA)
echo   🎯 Class-wise performance monitoring
echo   ⏰ Early stopping + gradient clipping
echo.
echo 🎯 EXPECTED PERFORMANCE:
echo   Validation Dice: 0.58+ (optimized hyperparameters from Trial #281)
echo   Stable 32-bit training without mixed precision issues
echo.
echo ⏱️  Training Duration: 120 epochs (optimal for OneCycleLR)
echo 📊 Progressive Resizing: 128px → 256px (full size at epoch 84)
echo 🛑 Early Stopping: 30 epochs patience (25% of total)
echo 🔄 SWA Start: Epoch 60 (50% of training)
echo.
echo 📚 Using virtual environment Python...
echo.

venv\Scripts\python.exe training/train.py ^
    --lr 2.11e-4 ^
    --dropout 0.09 ^
    --batch_size 16 ^
    --epochs 120 ^
    --weight_decay 1.04e-3 ^
    --warmup_epochs 5 ^
    --early_stopping_patience 30 ^
    --grad_clip_val 1.83 ^
    --use_swa ^
    --swa_start_epoch 60

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