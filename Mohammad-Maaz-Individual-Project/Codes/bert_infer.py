from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('model/checkpoint-12000', local_files_only=True)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

model.to(device)

import pandas as pd

comments_df = pd.read_csv('cryptocurrency_comments_cleaned_subset.csv')

comments_list = [text for text in comments_df['body']]


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('model/checkpoint-12000', local_files_only=True)
comments_encodings = tokenizer(comments_list, truncation=True)


import torch
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['attention_mask'])

comments_dataset = CustomDataset(comments_encodings)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from torch.utils.data import DataLoader
comments_dataloader = DataLoader(comments_dataset, batch_size=128, collate_fn=data_collator)

from tqdm.auto import tqdm
progress_bar = tqdm(range(len(comments_dataloader)))

predicted_ids = []
for batch in comments_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predicted_ids.extend(torch.argmax(outputs.logits, dim=-1).cpu().detach().numpy().tolist())
    progress_bar.update(1)

label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
id2label = {idx:label for idx,label in enumerate(label_names)}
predicted_labels = [id2label[id] for id in predicted_ids]
comments_df['emotion'] = predicted_labels
comments_df.to_csv('cryptocurrency_comments_cleaned_subset_emotion.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='darkgrid')
emotion_counts = comments_df['emotion'].value_counts().reset_index()
emotion_counts = emotion_counts.rename(columns={'index':'emotion', 'emotion': 'percentage_comments'})
emotion_counts['percentage_comments'] = 100*emotion_counts['percentage_comments']/comments_df.shape[0]
sns.barplot(data=emotion_counts, x='emotion', y='percentage_comments')
plt.tight_layout()
plt.show()