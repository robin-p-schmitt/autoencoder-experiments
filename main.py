from pipelines.ff_uc_autoencoder.main import run_shallow_ff_uc_ae, run_deep_ff_uc_ae
from pipelines.conv_uc_autoencoder.main import run_conv_uc_ae


def main():
  # run_shallow_ff_uc_ae()
  # run_deep_ff_uc_ae()
  run_conv_uc_ae()


if __name__ == '__main__':
  main()
