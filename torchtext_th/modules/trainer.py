from typing import List, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data as utils
from ignite.engine import Events, create_supervised_trainer, \
    create_supervised_evaluator
from ignite.metrics import Loss
from ignite.handlers import EarlyStopping
from ignite.engine.engine import Engine
from ignite._utils import convert_tensor
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter

from torchtext_th.data.vocab import Vocab
from torchtext_th.data.label import Label
from torchtext_th.data.sentence import Sentence
from torchtext_th.utils import get_params, get_summary_writer, model_summary


class Trainer(object):

    def __init__(self, model: nn.Module, vocab: Vocab, label: Label,
                 max_len: int = 200, log_dir: Optional[str] = None,
                 norm_char: bool = True):
        self.model = model
        self.vocab = vocab
        self.label = label
        self.max_len = max_len
        self.log_dir = log_dir
        self.norm_char = norm_char
        self.writer: Optional[SummaryWriter] = None

    def _on_epoch_complete(self, trainer_engine: Engine, evaluator: Engine,
                           data_loader: utils.DataLoader):
        pass

    def _on_iteration_complete_test(self, trainer_engine: Engine):
        iter_num = trainer_engine.state.iteration
        epoch = trainer_engine.state.epoch

        loss = trainer_engine.state.output
        self.writer.add_scalar("training/loss", loss, iter_num)
        if iter_num % 10 == 0:
            print(f"{str(datetime.now())} [{epoch}][{iter_num}] -- [{loss}]")

    def _on_iteration_complete_val(self, trainer_engine: Engine,
                                   evaluator: Engine,
                                   val_data_loader: utils.DataLoader):
        iter_num = trainer_engine.state.iteration
        epoch = trainer_engine.state.epoch
        if iter_num % 50 == 0:
            # Write evaluation loss
            evaluator.run(val_data_loader)
            metrics = evaluator.state.metrics
            avg_nll = metrics['nll']
            self.writer.add_scalar("validation/loss", avg_nll, iter_num)
            print(f"Validation Results - Epoch: {epoch} - Iter: {iter_num} "
                  f"Avg loss: {avg_nll:.5f}")

    def _on_iteration_complete_save(self, trainer_engine: Engine,
                                    model_saver: ModelCheckpoint):
        iter_num = trainer_engine.state.iteration - 1
        if iter_num % 500 == 0:
            to_save = {'model': self.model}
            model_saver(trainer_engine, to_save)
            print("Model checkpoint is created")


    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def get_data_loader(self, sentences: List[Sentence],
                        max_len: int, batch_size: int,
                        shuffle: bool = True) -> utils.DataLoader:
        char_vecs: List[List[str]] = []
        label_vecs: List[List[str]] = []
        for sentence in sentences:
            chars = list(sentence.to_chars(self.norm_char))
            char_vecs.append(chars)
            label_vecs.append(list(sentence.to_bmes_labels()))

        x_tensor: torch.Tensor = self.vocab.transform(chars=char_vecs,
                                                      max_len=max_len)
        x_len_tensor: torch.Tensor = (x_tensor != 0).sum(dim=1)
        y_tensor: torch.Tensor = self.label.transform(labels=label_vecs,
                                                      max_len=max_len)

        dataset = utils.TensorDataset(x_tensor, x_len_tensor, y_tensor)
        data_loader = utils.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle)
        return data_loader

    def _register_early_stopping(self, loss_fn, device, trainer):
        prepare_batch = __class__._prepare_batch
        evaluator = create_supervised_evaluator(model=self.model,
                                                metrics={'nll': Loss(loss_fn)},
                                                device=device,
                                                prepare_batch=prepare_batch)

        def score_fn(engine):
            return - engine.state.metrics['nll']

        early_stopping = EarlyStopping(patience=5,
                                       score_function=score_fn,
                                       trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, early_stopping)
        return evaluator

    @staticmethod
    def _prepare_batch(batch, device=None, non_blocking=False):
        """
        Prepare batch for training: pass to a device with options.
        """
        x, x_len, y = batch
        x = convert_tensor(x, device=device, non_blocking=non_blocking)
        x_len = convert_tensor(x_len, device=device, non_blocking=non_blocking)
        y = convert_tensor(y, device=device, non_blocking=non_blocking)

        sorted_len, perm_idx = x_len.sort(0, descending=True)
        x = x[perm_idx]
        y = y[perm_idx]

        input_dict = dict(
            x=x,
            x_length=sorted_len,
            is_training=True
        )
        return input_dict, y

    def fit(self, train_sentences: List[Sentence], batch_size: int,
            epochs: int, lr: float = 0.001,
            val_sentences: Optional[List[Sentence]] = None,
            checkpoint_dir: Optional[str] = None):

        print(model_summary(self.model))

        self.writer = get_summary_writer(log_dir=self.log_dir)

        train_data_loader = self.get_data_loader(train_sentences,
                                                 max_len=self.max_len,
                                                 batch_size=batch_size)
        print("Training data size", len(train_data_loader.dataset))

        device = __class__.get_device()
        loss_fn = self.model.loss

        trainable_params = get_params(self.model, requires_grad=True)
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        prepare_batch = __class__._prepare_batch
        trainer = create_supervised_trainer(model=self.model,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            device=device,
                                            prepare_batch=prepare_batch)
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  self._on_iteration_complete_test)

        #############################
        # Set up validation process #
        #############################
        if val_sentences is not None:
            val_dataloader = self.get_data_loader(val_sentences,
                                                  max_len=self.max_len,
                                                  batch_size=batch_size)
            evaluator = self._register_early_stopping(loss_fn, device, trainer)
            trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                      self._on_iteration_complete_val,
                                      evaluator,
                                      val_dataloader)

        ############################
        # Set up model check-point #
        ############################
        if checkpoint_dir is not None:
            print(f"Model checkpoint will be stored at {checkpoint_dir}")
            handler = ModelCheckpoint(dirname=checkpoint_dir,
                                      filename_prefix="",
                                      save_interval=1,
                                      n_saved=10,
                                      create_dir=True,
                                      atomic=True,
                                      save_as_state_dict=True)
            trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                      self._on_iteration_complete_save,
                                      handler)

        trainer.run(train_data_loader, max_epochs=epochs)

        print("Finish training ....")
        return self
