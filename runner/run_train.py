# for SFT, pre-training, dpo
import logging
import sys

from transformers import Trainer, Seq2SeqTrainer
from trl import DPOTrainer

sys.path.append("../..")
from llmx.args.parser import parse_args
from llmx.model.model_loader import prepare_model
from llmx.data.data_loader import prepare_data
from llmx.utils.patches.dpo_trainer_patch import patch_dpo_trainer

logging.basicConfig(level=logging.INFO)


def get_trainer(finetuning_args):
    stage = finetuning_args.training_stage
    if stage == "pt":  # pretrain
        return Trainer

    elif stage == "sft":  # supervised finetuning
        return Seq2SeqTrainer

    elif stage == "dpo":  # dpo
        patch_dpo_trainer(DPOTrainer)
        return DPOTrainer
    else:
        raise Exception(f"training stage: {stage} is not supported for now.")


# 1. parse args
training_args, data_args, finetuning_args, generating_args, model_args, \
    peft_args = parse_args(on_train=True)

# 2. prepare model, tokenizer, etc,.
ref_model, model, tokenizer = prepare_model(
    model_args, training_args, peft_args, finetuning_args,
)

# 3. prepare data
train_dataset, eval_dataset, data_collator = prepare_data(
    model_args, data_args, finetuning_args, tokenizer,
)

trainer_class = get_trainer(finetuning_args)

if finetuning_args.training_stage == "dpo":
    ext_args = {
        "ref_model": ref_model,
        "beta": finetuning_args.dpo_beta,
        "loss_type": finetuning_args.dpo_loss_type,
    }
else:
    ext_args = {}
    
# 4. prepare trainer
trainer = trainer_class(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    tokenizer=tokenizer,
    callbacks=[],
    **ext_args,
)

# 5. train / save / evaluate
if training_args.do_train:
    result = trainer.train()
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    
if training_args.do_eval:
    metrics = trainer.evaluate(metric_key_prefix="eval", **generating_args)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
if training_args.do_predict:
    results = trainer.evaluate(metric_key_prefix="predict", **generating_args)
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
    trainer.save_predictions(results)
