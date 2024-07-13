# 基于预训练加微调的机器阅读理解问答系统

## 项目介绍

本项目使用中文[RoBERTa](https://huggingface.co/uer/roberta-base-chinese-extractive-qa)预训练模型，用[CMRC 2018](https://github.com/ymcui/cmrc2018/tree/master)数据集对模型进行微调，然后针对给定的中山大学介绍性中文文本，使用带有重叠的滑动窗口法实现片段抽取式的机器阅读理解问答系统

## 环境配置

```
accelerate=0.21.0
datasets=2.20.0
nltk=3.8.1
numpy=1.24.4
python=3.8.19
torch=2.3.0+cu121
torch-scatter=2.0.8
torchaudio=2.3.0+cu121
torchvision=0.18.0+cu121
tqdm=4.66.4
transformers=4.36.2
```
### nltk的相关包下载
* 首先要下载nltk
  ```sh
  pip install nltk
  ```
* 然后要下载punkt
  [下载punkt](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip)并将解压好的`punkt`文件夹放到到下面路径里面的其中一个即可
  
    ```
    - '/public/home/user/nltk_data/tokenizers'                                           
    - '/public/home/user/anaconda3/envs/env_name/nltk_data/tokenizers'                              
    - '/public/home/user/anaconda3/envs/env_name/share/nltk_data/tokenizers'                                       - '/public/home/user/anaconda3/envs/env_name/lib/nltk_data/tokenizers'                                       
    - '/usr/share/nltk_data/tokenizers'                                                   
    - '/usr/local/share/nltk_data/tokenizers'                                             
    - '/usr/lib/nltk_data/tokenizers'                                                     
    - '/usr/local/lib/nltk_data/tokenizers'                                                            
    ```

## 代码介绍

```
│  cmrc_eval.py                          # 内含CMRC评估函数
│  QA_main.py                            # 运行问答系统
│  README.md                             
│  train.py                              # 模型微调
│  train_data_sysu_Chinese.txt           # 答案的来源文本
│  
├─model                                  # 存放预训练模型
│      README.md
│      
├─models_for_qa                          # 存放微调过后的模型
│      README.md
│      
└─mrc_data                               # 存放CMRC数据集
    │  dataset_dict.json
    │  
    ├─test
    │      data-00000-of-00001.arrow
    │      dataset_info.json
    │      state.json
    │      
    ├─train
    │      data-00000-of-00001.arrow
    │      dataset_info.json
    │      state.json
    │      
    └─validation
            data-00000-of-00001.arrow
            dataset_info.json
            state.json
```

## 代码运行

* 预训练模型的微调

  ```sh
  pyhton train.py
  ```
* 运行问答系统
  ```sh
  python QA_main.py
  ```

## 参考材料

* [zyds/transformers-code (github.com)](https://github.com/zyds/transformers-code)

* [ymcui/cmrc2018: A Span-Extraction Dataset for Chinese Machine Reading Comprehension (CMRC 2018) (github.com)](https://github.com/ymcui/cmrc2018/tree/master)

* [dbiir/UER-py: Open Source Pre-training Model Framework in PyTorch & Pre-trained Model Zoo (github.com)](https://github.com/dbiir/UER-py/)
