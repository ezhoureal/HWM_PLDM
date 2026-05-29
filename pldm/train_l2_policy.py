import sys

from pldm.train_latent_policy import main


if __name__ == "__main__":
    main(["--policy_level", "l2", *sys.argv[1:]])
