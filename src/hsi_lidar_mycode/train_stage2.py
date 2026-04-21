import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated combined Stage2 trainer. Use dedicated single-missing trainers."
    )
    parser.parse_args()
    raise SystemExit(
        "Combined Stage2 training is removed.\n"
        "Use one of:\n"
        "  python src/stage1_hsi_lidar/train_stage2_missing_hsi.py\n"
        "  python src/stage1_hsi_lidar/train_stage2_missing_lidar.py"
    )


if __name__ == "__main__":
    main()
