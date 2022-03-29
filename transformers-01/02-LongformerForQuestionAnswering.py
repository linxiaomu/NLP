from transformers import LongformerTokenizer, LongformerForQuestionAnswering
import torch

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
# question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
# question, text = "What does Jim Henson like?", "Jim Henson likes to eat apples"
# question, text = "小明中午吃的什么?", "小明早饭吃的馅饼，午饭吃的文苑二楼的犀米饭，晚上想去吃火锅"
question, text = "What did Xiao Ming have for lunch?", "Xiao Ming had pie for breakfast and rhinoceros rice for lunch on the second floor of Wen Yuan. He wanted to eat hot pot in the evening"
encoding = tokenizer(question, text, return_tensors="pt")
input_ids = encoding["input_ids"]
# default is local attention everywhere
# the forward method will automatically set global attention on question tokens
attention_mask = encoding["attention_mask"]
outputs = model(input_ids, attention_mask=attention_mask)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
answer_tokens = all_tokens[torch.argmax(start_logits):torch.argmax(end_logits) + 1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))  # remove space prepending space token
print(answer)
