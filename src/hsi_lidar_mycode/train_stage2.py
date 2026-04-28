import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated combined trainer. Use Stage2 mapper pretraining and Stage3 diffusion trainers."
    )
    parser.parse_args()
    raise SystemExit(
        "Combined training is removed.\n"
        "Use:\n"
        "  python src/hsi_lidar_mycode/train_stage2_mapper.py --stage1_ckpt <stage1_best.pt>\n"
        "Then one of:\n"
        "  python src/hsi_lidar_mycode/train_stage3_missing_hsi.py --stage1_ckpt <stage1_best.pt> --stage2_mapper_ckpt <stage2_best.pt>\n"
        "  python src/hsi_lidar_mycode/train_stage3_missing_lidar.py --stage1_ckpt <stage1_best.pt> --stage2_mapper_ckpt <stage2_best.pt>"
    )


if __name__ == "__main__":
    main()
