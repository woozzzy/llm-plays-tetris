import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

with open("training_data.json", "r") as f:
    game_states = json.load(f)

with open("labels.json", "r") as f:
    labels = json.load(f)

frame_to_state = {state["frame"]: state for state in game_states["states"]}

frame_to_actions = {}

for event in labels["events"]:
    frame = event["frame"]
    event_type = event["type"]
    data = event["data"]
    if event_type in ["keydown", "keyup"]:
        action = f"{event_type}_{data['key']}"
        if frame in frame_to_actions:
            frame_to_actions[frame].append(action)
        else:
            frame_to_actions[frame] = [action]

dataset = []

for state in game_states["states"]:
    frame = state["frame"]
    game_board = state["game_board"]
    current_piece = state["current_piece"]
    held_piece = state["held_piece"]
    can_hold = state["can_hold"]
    next_pieces = state["next_pieces"]

    state_text = f"Frame: {frame}\n"
    state_text += "Game Board:\n"
    for row in game_board:
        state_text += "".join(row) + "\n"
    state_text += f"Current Piece: {current_piece}\n"
    state_text += f"Held Piece: {held_piece}\n"
    state_text += f"Can Hold: {can_hold}\n"
    state_text += f"Next Pieces: {' '.join(next_pieces)}\n"

    actions = frame_to_actions.get(frame, [])
    if actions:
        action_text = " ".join(actions)
        dataset.append({"input": state_text, "output": action_text})

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

special_tokens_dict = {"additional_special_tokens": ["<INPUT>", "<OUTPUT>"]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


def preprocess(example):
    text = "<INPUT>\n" + example["input"] + "\n<OUTPUT>\n" + example["output"]
    encoding = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    input_ids = encoding["input_ids"]
    labels = input_ids.copy()

    output_token_id = tokenizer.convert_tokens_to_ids("<OUTPUT>")
    output_token_index = input_ids.index(output_token_id)

    labels[: output_token_index + 1] = [-100] * (output_token_index + 1)

    encoding["labels"] = labels
    return encoding


dataset_hf = Dataset.from_list(dataset)

dataset_tokenized = dataset_hf.map(preprocess, remove_columns=["input", "output"])

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,
    dataloader_num_workers=4,
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tokenized,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./fine_tuned_model")
