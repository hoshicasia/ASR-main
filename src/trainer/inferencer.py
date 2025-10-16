from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        logger=None,
        writer=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            logger (Logger | None): logger for logging inference progress.
            writer (Writer | None): experiment tracker writer for logging
                predictions, metrics, and other artifacts.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder
        self.logger = logger
        self.writer = writer

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=writer,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def _decode_predictions(self, log_probs, log_probs_length):
        """
        Decode log probabilities into text predictions using configured method.

        Args:
            log_probs (Tensor): Log probabilities of shape [batch_size, seq_len, vocab_size].
            log_probs_length (Tensor): Length of each sequence in the batch.

        Returns:
            list[str]: Decoded text predictions for each sample in the batch.
        """
        decode_method = self.cfg_trainer.get("decode_method", "argmax")

        if decode_method == "beam_search":
            return self._decode_beam_search(log_probs, log_probs_length)
        elif decode_method == "beam_search_lm":
            return self._decode_beam_search_lm(
                log_probs,
                log_probs_length,
                beam_width=self.cfg_trainer.get("beam_width", 50),
                alpha=self.cfg_trainer.get("lm_alpha", 0.5),
                beta=self.cfg_trainer.get("lm_beta", 0.0),
            )
        else:
            return self._decode_argmax(log_probs, log_probs_length)

    def _decode_argmax(self, log_probs, log_probs_length):
        """
        Decode using greedy argmax approach.

        Args:
            log_probs (Tensor): Log probabilities of shape [batch_size, seq_len, vocab_size].
            log_probs_length (Tensor): Length of each sequence in the batch.

        Returns:
            list[str]: Decoded text predictions.
        """
        argmax_inds = log_probs.argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        return [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

    def _decode_beam_search(self, log_probs, log_probs_length, beam_width=None):
        """
        CTC Beam Search
        Args:
            log_probs (Tensor): Log probabilities of shape [batch_size, seq_len, vocab_size].
            log_probs_length (Tensor): Length of each sequence in the batch.
            beam_width (int | None): Beam width for the search. If None, uses default from config.
        Returns:
            list[str]: Decoded text predictions.
        """
        import numpy as np

        if beam_width is None:
            beam_width = int(self.cfg_trainer.get("beam_width", 10))
        lp = log_probs.detach().cpu().numpy()
        lens = log_probs_length.detach().cpu().numpy().astype(int)

        blank = 0
        B = lp.shape[0]
        out = []

        for b in range(B):
            T = int(lens[b])
            if T == 0:
                out.append("")
                continue

            probs = lp[b, :T, :]
            V = probs.shape[1]
            beams = {(): (0.0, -np.inf)}

            for t in range(T):
                lpt = probs[t]
                next_beams = {}
                topk = np.argsort(-lpt)[: min(V, beam_width + 5)]

                for pref, (pb, pnb) in beams.items():
                    total = np.logaddexp(pb, pnb)

                    nb_pb = total + float(lpt[blank])
                    prev = next_beams.get(pref, (-np.inf, -np.inf))
                    next_beams[pref] = (np.logaddexp(prev[0], nb_pb), prev[1])

                    for k in topk:
                        k = int(k)
                        if k == blank:
                            continue
                        new_pref = pref + (k,)
                        add = (
                            pb + float(lpt[k])
                            if pref and pref[-1] == k
                            else total + float(lpt[k])
                        )
                        prev = next_beams.get(new_pref, (-np.inf, -np.inf))
                        next_beams[new_pref] = (prev[0], np.logaddexp(prev[1], add))

                scored = [
                    (pfx, s, np.logaddexp(s[0], s[1])) for pfx, s in next_beams.items()
                ]
                scored.sort(key=lambda x: x[2], reverse=True)
                beams = {pfx: s for pfx, s, _ in scored[:beam_width]}

            best = max(beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))[
                0
            ]
            decoded = self.text_encoder.ctc_decode(list(best))
            out.append(decoded)
        return out

    def _decode_beam_search_lm(
        self,
        log_probs,
        log_probs_length,
        beam_width=1500,
        alpha=3.23,
        beta=-0.26,
        kenlm_files_name="librispeech-4-gram",
    ):
        """
        Decode using beam search with language model.
        """

    def _save_predictions(self, predictions, audio_paths, part):
        """
        Save predictions to text files.

        Args:
            predictions (list[str]): Text predictions to save.
            audio_paths (list[str]): Corresponding audio file paths.
            part (str): Dataset partition name (used for directory structure).
        """
        for prediction_text, audio_path in zip(predictions, audio_paths):
            utterance_id = Path(audio_path).stem
            output_file = self.save_path / part / f"{utterance_id}.txt"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with output_file.open("w", encoding="utf-8") as f:
                f.write(prediction_text)

    def _log_predictions(self, predictions, targets, audio_paths, part):
        """
        Log predictions to experiment tracker (CometML/TensorBoard).

        Args:
            predictions (list[str]): Text predictions.
            targets (list[str]): Ground truth texts.
            audio_paths (list[str]): Corresponding audio file paths.
            part (str): Dataset partition name.
        """
        if self.writer is None:
            return

        examples_to_log = self.cfg_trainer.get("examples_to_log", 20)

        tuples = list(zip(predictions, targets, audio_paths))
        rows = {}

        for pred, target, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer = self._calc_wer(target, pred) * 100
            cer = self._calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "prediction": pred,
                "WER": f"{wer:.1f}",
                "CER": f"{cer:.1f}",
            }

        import pandas as pd

        self.writer.add_table(
            f"predictions_{part}", pd.DataFrame.from_dict(rows, orient="index")
        )

    def _calc_wer(self, target, prediction):
        """Calculate Word Error Rate."""
        from src.metrics.utils import calc_wer

        return calc_wer(target, prediction)

    def _calc_cer(self, target, prediction):
        """Calculate Character Error Rate."""
        from src.metrics.utils import calc_cer

        return calc_cer(target, prediction)

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        log_probs = batch["log_probs"].cpu()
        log_probs_length = batch["log_probs_length"].cpu()

        predictions = self._decode_predictions(log_probs, log_probs_length)
        batch["predictions"] = predictions
        batch_size = log_probs.shape[0]
        audio_paths = batch.get("audio_path", [None] * batch_size)

        # Save predictions to files
        if self.save_path is not None:
            self._save_predictions(predictions, audio_paths, part)

        # Store predictions for logging
        batch["predictions"] = predictions
        batch["prediction_audio_paths"] = audio_paths

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        # Collect predictions for logging
        all_predictions = []
        all_targets = []
        all_audio_paths = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

                # Collect data for logging
                if self.writer is not None:
                    all_predictions.extend(batch.get("predictions", []))
                    all_targets.extend(batch.get("text", []))
                    all_audio_paths.extend(batch.get("prediction_audio_paths", []))

        results = self.evaluation_metrics.result()
        if self.writer is not None:
            self.writer.add_scalars(results)
        if self.logger:
            for key, value in results.items():
                self.logger.info(f"  {key}: {value}")

        # Log predictions to experiment tracker
        if self.writer is not None and len(all_predictions) > 0:
            self.writer.set_step(0, mode=f"inference_{part}")
            self._log_predictions(all_predictions, all_targets, all_audio_paths, part)
            if self.logger:
                self.logger.info(
                    f"Logged {len(all_predictions)} predictions to experiment tracker"
                )

        return results
