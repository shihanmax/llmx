# for SFT, pre-training, dpo
import logging
import sys


sys.path.append("../..")
from llmx.args.parser import parse_args
from llmx.data.data_loader import prepare_data
from llmx.model.model_loader import ModelLoader 
from llmx.trainer.trainer import prepare_trainer


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

# 1. parse args
training_args, data_args, finetuning_args, generating_args, model_args, \
    peft_args = parse_args(on_train=True)

# 2. prepare model, tokenizer, etc,.
ref_model, model, tokenizer = ModelLoader.load(
    model_args, training_args, finetuning_args, data_args, peft_args, 
)

# 3. prepare data
train_dataset, eval_dataset, data_collator = prepare_data(
    model_args, data_args, finetuning_args, tokenizer,
)

# 4. prepare trainer
trainer = prepare_trainer(
    finetuning_args, training_args, model_args, model, ref_model, tokenizer, 
    train_dataset, eval_dataset, data_collator,
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
