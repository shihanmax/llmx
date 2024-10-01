import logging

import datasets
import torch

from typing import Any, Dict, Union

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, Seq2SeqTrainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
from trl import DPOTrainer

from ..utils.patches.sequence_parallel import (
    SequenceParallelSampler, get_sequence_parallel_world_size,
    init_sequence_parallel, get_sequence_parallel_group,
    split_for_sequence_parallel, reduce_sequence_parallel_loss,
    get_rank,
)
from ..utils.patches.dpo_trainer_patch import patch_dpo_trainer
from ..utils.gutil_fast import get_avg_device_utilization

logger = logging.getLogger(__name__)


class SFTTrainer(Seq2SeqTrainer):

    """Custom trainer for SFT"""
    def _get_train_sampler(self):
        """override for sequence parallel"""
        if get_sequence_parallel_world_size() > 1:
            return SequenceParallelSampler(self.train_dataset)
        
        return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if get_sequence_parallel_world_size() > 1:
            return DataLoader(train_dataset, **dataloader_params)
        else:
            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def record_device_utilization(self):
        if self.state.is_local_process_zero and \
            self.state.global_step % 10 == 0:  # record gpu info every 10 steps
                info = get_avg_device_utilization()
                self.log(info)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        outputs = super().training_step(model, inputs)

        self.record_device_utilization()
        return outputs
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """override for sequence parallel"""
        if get_sequence_parallel_world_size() > 1:
            return self._compute_loss_sequence_parallel(
                model, inputs, return_outputs,
            )
        else:
            return super().compute_loss(model, inputs, return_outputs)
    
    def debug_tensor(self, data, msg):
        rank = get_rank()
        for k, v in data.items():
            logger.debug(f"|--{msg}--> rank: {rank}: {k}, {v.shape}")
            if k == "input_ids":
                try:
                    logger.debug(f"|--{msg}--> rank: {rank}: {self.tokenizer.batch_decode(v)}")
                except:
                    logger.error(f"|--{msg}--> rank: {rank}: batch_decode failed, input_ids: {input_ids}")

    def _split_for_sequence_parallel(self, data):
        # not split for attention mask
        sp_group = get_sequence_parallel_group()

        for key in ('input_ids', 'labels', 'position_ids'):
            val = data.get(key, None)
            if val is not None:
                # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                data[key] = split_for_sequence_parallel(
                    val, dim=1, sp_group=sp_group,
                )

        return data

    def _compute_loss_sequence_parallel(self, model, inputs, return_outputs):
        """compute loss for sequence parallel"""

        inputs = self._split_for_sequence_parallel(inputs)
    
        outputs = model(**inputs)

        labels = inputs["labels"]
        num_tokens = (labels != -100).sum()

        sp_group = get_sequence_parallel_group()
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = reduce_sequence_parallel_loss(
            loss, loss_scale=num_tokens, sp_group=sp_group,
        )
        
        return (loss, outputs) if return_outputs else loss


def prepare_trainer(
    finetuning_args, training_args, model_args, model, ref_model, tokenizer, 
    train_dataset, eval_dataset, data_collator, 
):
    if finetuning_args.training_stage == "dpo":
        trainer_ext_kwargs = {
            "ref_model": ref_model,
            "beta": finetuning_args.dpo_beta,
            "loss_type": finetuning_args.dpo_loss_type,
        }
    else:
        trainer_ext_kwargs = {}

    stage = finetuning_args.training_stage
    
    if stage == "pt":  # pretrain
        trainer_cls = Trainer

    elif stage == "sft":  # supervised finetuning
        trainer_cls = SFTTrainer

    elif stage == "dpo":  # dpo
        patch_dpo_trainer(DPOTrainer)
        trainer_cls = DPOTrainer
        
    else:
        raise Exception(f"training stage: {stage} is not supported for now.")

    trainer = trainer_cls(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[],
        **trainer_ext_kwargs,
    )
    
    if model_args.sequence_parallel_size > 1:
        logger.info("init sequence parallel env")
        init_sequence_parallel(model_args.sequence_parallel_size)

    return trainer
