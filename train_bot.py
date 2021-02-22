import argparse
from typing import Optional

from transformers import GPT2Tokenizer, GPT2LMHeadModel, LineByLineTextDataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


class Config:
    output_dir: str
    input_model: Optional[str]
    input_file: str


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="Where to store the model ?")
    parser.add_argument("--input_file", help="The input file. Each record must be on its line")
    parser.add_argument("--input_model",
                        help="the folder of a previously trained model. If you want to continue training it", default=None)

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.input_model is None:
        model = GPT2LMHeadModel.from_pretrained("antoiloui/belgpt2")
    else:
        print('loading pre trained model')
        model = GPT2LMHeadModel.from_pretrained(args.input_model)

    tokenizer = GPT2Tokenizer.from_pretrained("antoiloui/belgpt2")

    training_args = TrainingArguments(
        output_dir=args.output_dir + '_checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs_hyca',  # directory for storing logs
        logging_steps=100,
    )
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
    tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))
    dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=args.input_file, block_size=32)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    if args.input_model is not None:
        trainer.train(resume_from_checkpoint=args.input_model + '_checkpoint')
    else:
        trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
