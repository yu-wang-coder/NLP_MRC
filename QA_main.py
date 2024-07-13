from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("./model")

# 加载微调好的模型，因为前面训练了5轮，所以加载最后一轮的checkpoint-550模型
model = AutoModelForQuestionAnswering.from_pretrained("models_for_qa/checkpoint-550")

# 加载中山大学中文简介数据并预处理
context = ''
with open('train_data_sysu_Chinese.txt', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        context += line.strip()

# 模型预测
from transformers import pipeline
pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)
# 设置了6个问题
q = ['中山大学起初校名叫什么?', '中山大学是由谁创办的？', '中山大学的校训词是什么？', '中山大学的校歌是由谁作词？', '中山大学有多久的办学历史？','中山大学是哪一年创办的？']
# 进行预测
for i in q:
    print(i)
    print(pipe(question=i, context=context))