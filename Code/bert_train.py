# load emotion dataset
from datasets import load_dataset
emotion_raw = load_dataset("emotion")

# tokenize using bert tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

emotion_tokenized = emotion_raw.map(tokenize_function, batched=True)

# import a datacollator for padding to the same size
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# load a pretrained bert model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=6)

# set the evaluation metric as f1 score and define it
from datasets import load_metric
import numpy as np
metric = load_metric("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')

# define training args
training_args = TrainingArguments(
    output_dir="model",
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    metric_for_best_model = 'f1',
    load_best_model_at_end=True
)

# define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=emotion_tokenized["train"],
    eval_dataset=emotion_tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# train model
trainer.train()

# plot loss graph
loss_list = [log['loss'] for log in trainer.state.log_history if len(log.keys())==4]
epoch_list = [log['epoch'] for log in trainer.state.log_history if len(log.keys())==4]
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# view eval f1s
import pandas as pd
pd.DataFrame([(log['epoch'], log['eval_f1'], log['eval_loss']) for log in trainer.state.log_history if 'eval_f1' in log], columns = ['epoch', 'eval_f1', 'eval_loss'])

# view eval performance
trainer.evaluate()

# evaluate on test data
predictions = trainer.predict(emotion_tokenized['test'])

predicted_labels = np.argmax(predictions.predictions, axis=-1)
target_labels = predictions.label_ids

from sklearn.metrics import classification_report

report = classification_report(target_labels, predicted_labels, target_names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])
print(report)