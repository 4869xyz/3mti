try:
    from .train_stage2_single_missing import run_training_for_mode
except ImportError:
    from train_stage2_single_missing import run_training_for_mode


if __name__ == "__main__":
    run_training_for_mode(
        missing_mode="lidar_missing",
        default_output_dir="outputs/stage2_houston_missing_lidar",
        description="Stage2 training for missing LiDAR only",
    )
