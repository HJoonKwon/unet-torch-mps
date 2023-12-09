import logging
import torch


def load_checkpoint(model, optimizer, ckpt_path, logger):
    if ckpt_path is not None:
        ckeckpoint = torch.load(ckpt_path)
        start_epoch = ckeckpoint["epoch"] + 1
        model.load_state_dict(ckeckpoint["model_state_dict"])
        optimizer.load_state_dict(ckeckpoint["optimizer_state_dict"])
        train_loss = ckeckpoint["train_loss"]
        valid_loss = ckeckpoint["valid_loss"]
        logger.info(
            f"Loaded ckpt from: {ckpt_path} @ epoch: {ckeckpoint['epoch']} with train_loss: {train_loss} and valid_loss: {valid_loss}"
        )
    else:
        start_epoch = 0
    return start_epoch


def set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
