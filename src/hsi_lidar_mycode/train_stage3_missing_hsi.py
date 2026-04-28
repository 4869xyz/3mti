try:
    from .train_stage2_single_missing import run_training_for_mode
except ImportError:
    from train_stage2_single_missing import run_training_for_mode


if __name__ == "__main__":
    run_training_for_mode(
        missing_mode="hsi_missing",
        default_output_dir="outputs/stage3_houston_missing_hsi",
        description="Stage3 diffusion training for missing HSI only",
    )
