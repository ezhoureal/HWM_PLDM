from pldm.train import main, TrainConfig


if __name__ == "__main__":
    cfg = TrainConfig.parse_from_command_line()
    main(cfg)
