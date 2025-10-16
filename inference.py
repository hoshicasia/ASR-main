import logging
import warnings
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.logger import CometMLWriter
from src.logger.logger import setup_logging
from src.metrics.utils import calc_cer, calc_wer
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    save_dir = (
        ROOT_PATH
        / config.inferencer.get("save_dir", "saved")
        / "inference"
        / config.inferencer.get("run_name", "default")
    )
    save_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(config, save_dir / "config.yaml")
    setup_logging(save_dir, append=False)
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    logger.info(model)

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        # use text_encoder in metrics
        metrics["inference"].append(
            instantiate(metric_config, text_encoder=text_encoder)
        )

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    # setup writer for logging (optional)
    writer = None
    if config.get("writer") is not None and config.inferencer.get("log_results", False):
        # Manually create writer with required arguments
        writer = CometMLWriter(
            logger=logger,
            project_config=config,
            project_name=config.writer.get("project_name", "asr_inference"),
            workspace=config.writer.get("workspace", None),
            run_name=config.writer.get("run_name", "inference"),
            mode=config.writer.get("mode", "online"),
        )
        logger.info("Writer initialized for logging inference results")

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        text_encoder=text_encoder,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        logger=logger,
        writer=writer,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    logger.info("=" * 50)
    logger.info("Inference Results:")
    logger.info("=" * 50)
    for part in logs.keys():
        logger.info(f"\n{part.upper()} partition:")
        for key, value in logs[part].items():
            full_key = part + "_" + key
            logger.info(f"    {full_key:15s}: {value}")

    logger.info("\nInference completed successfully!")


if __name__ == "__main__":
    main()
