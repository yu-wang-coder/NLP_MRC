import numpy as np
import collections
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator

# 加载数据集
datasets = DatasetDict.load_from_disk("mrc_data")

# 数据预处理
tokenizer = AutoTokenizer.from_pretrained("./model") # 加载分词器

def process_func(examples):
    """
    输入：原始数据集
    输出：重叠滑窗处理后的数据集

    tokenizer的参数含义
    text：表示将要拼接的“问题”文本
    text_pair：表示将要拼接的“题干”长文本
    return_offsets_mapping：表示返回token_id所对应的实际字符的位置区间
    return_overflowing_tokens：表示对过长的context的token进行重叠滑窗
    stride：表示滑窗进行128个token的重叠
    max_length：表示输入的最大文本长度，超过的要截断
    truncation：表示仅对第二部分文本进行截断，即仅对text_pair的内容进行截断
    padding：表示对短文本填充到max_length
    """
    tokenized_examples = tokenizer(text=examples["question"],
                               text_pair=examples["context"],
                               return_offsets_mapping=True,
                               return_overflowing_tokens=True,
                               stride=128,
                               max_length=512, truncation="only_second", padding="max_length")
    
    # 取出重叠滑窗处理后的token
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # 存储每个答案的token的起始和结束位置
    start_positions = []
    end_positions = []

    # 存储每一个数据的标识
    example_ids = []
    for idx, _ in enumerate(sample_mapping):
        answer = examples["answers"][sample_mapping[idx]] # 获取对应idx的答案
        start_char = answer["answer_start"][0] # 答案在原字符序列的起始位置
        end_char = start_char + len(answer["text"][0]) # 答案在原字符序列的结束位置

        # 定位答案在token中的起始位置和结束位置
        # 我们要拿到context的起始和结束，然后从左右两侧向答案逼近
        context_start = tokenized_examples.sequence_ids(idx).index(1) # 获取context的token的起始位置
        context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1 # 获取context的token的结束位置
        offset = tokenized_examples.get("offset_mapping")[idx] # 获取context的token所映射到的字符串区间

        # 判断答案是否在context中
        if offset[context_end][1] < start_char or offset[context_start][0] > end_char: # 两种情况均表示答案不在context中
            start_token_pos = 0 # 没有答案就定位到CLS上
            end_token_pos = 0
        else: # 如果有答案
            # 从左到右定位答案的起始token位置
            token_id = context_start 
            while token_id <= context_end and offset[token_id][0] < start_char:
                token_id += 1
            start_token_pos = token_id

            # 从右到左定位答案的结束token位置
            token_id = context_end
            while token_id >= context_start and offset[token_id][1] > end_char:
                token_id -=1
            end_token_pos = token_id

        start_positions.append(start_token_pos)
        end_positions.append(end_token_pos)
        example_ids.append(examples["id"][sample_mapping[idx]])

        # 将分隔符和问题部分的offset_mapping设置为None，只保留context的内容 
        tokenized_examples["offset_mapping"][idx] = [
            (o if tokenized_examples.sequence_ids(idx)[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][idx])
        ]

    
    tokenized_examples["example_ids"] = example_ids
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

# 数据集映射到滑窗处理后的数据集
tokenized_datasets = datasets.map(process_func, batched=True, remove_columns=datasets["train"].column_names)

# 获取模型的输出
def get_result(start_logits, end_logits, exmaples, features):
    """
    输入：起始位置得分，结束位置得分，原始数据集，重叠滑窗的数据集
    输出：预测的答案，真实的答案
    """

    predictions = {} # 存储预测值
    references = {} # 存储真实值

    # 获取数据集中每条数据id所对应的全部滑窗处理后的数据
    example_to_feature = collections.defaultdict(list)
    for idx, example_id in enumerate(features["example_ids"]):
        example_to_feature[example_id].append(idx)

    # 最优答案候选数
    n_best = 20

    # 最大答案长度
    max_answer_length = 30

    # 遍历每条数据
    for example in exmaples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # 遍历每条数据的所有滑窗数据
        for feature_idx in example_to_feature[example_id]:
            # 获取相应的答案的起始位置和结束位置的得分，以及字符串位置映射
            start_logit = start_logits[feature_idx]
            end_logit = end_logits[feature_idx]
            offset = features[feature_idx]["offset_mapping"]

            # 获取前n_best个得分高的结果
            start_indexes = np.argsort(start_logit)[::-1][:n_best].tolist()
            end_indexes = np.argsort(end_logit)[::-1][:n_best].tolist()

            # 对每个结果判断答案是否存在
            for start_index in start_indexes:
                for end_index in end_indexes:

                    # 答案落在问题处或分隔符处，则答案不存在
                    if offset[start_index] is None or offset[end_index] is None:
                        continue
                    
                    # 答案的起始位置大于结束位置，或答案的长度大于max_answer_length，则答案不存在
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    # 获取答案的文本＆得分
                    answers.append({
                        "text": context[offset[start_index][0]: offset[end_index][1]],
                        "score": start_logit[start_index] + end_logit[end_index]
                    })

        # 看最终是否有答案，记录到predictions里
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["score"])
            predictions[example_id] = best_answer["text"]
        else:
            predictions[example_id] = ""
        references[example_id] = example["answers"]["text"]

    return predictions, references

# 评估函数
# 导入CMRC 2018官方的评估函数
from cmrc_eval import evaluate_cmrc
def metirc(pred):
    start_logits, end_logits = pred[0]

    # 根据数据长度判断是在验证集还是测试集上评估
    if start_logits.shape[0] == len(tokenized_datasets["validation"]):
        p, r = get_result(start_logits, end_logits, datasets["validation"], tokenized_datasets["validation"])
    else:
        p, r = get_result(start_logits, end_logits, datasets["test"], tokenized_datasets["test"])

    return evaluate_cmrc(p, r)

# 加载模型
model = AutoModelForQuestionAnswering.from_pretrained("./model")

# 配置训练参数
"""
output_dir：微调模型的输出地址
per_device_train_batch_size：训练的批大小
per_device_eval_batch_size：评估的批大小
evaluation_strategy="steps"：按迭代批次数进行评估
eval_steps=200：每训练200批进行一次评估
save_strategy="epoch"：每轮保存一次参数
logging_steps=50：每50批保存一次日志
num_train_epochs：训练轮次
"""
args = TrainingArguments(
    output_dir="models_for_qa",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    logging_steps=50,
    num_train_epochs=5
)

# 设置训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DefaultDataCollator(),
    compute_metrics=metirc
)

# 模型训练
trainer.train()
