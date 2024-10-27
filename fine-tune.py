from transformers import LlamaForCausalLM, Trainer, TrainingArguments


def run():
    # Load the model
    model = LlamaForCausalLM.from_pretrained("llama-3-tiny")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    run()
