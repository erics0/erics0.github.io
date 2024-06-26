---
title: "Pipeline的工作流程"
subtitle: "Pipeline的工作流程"
layout: post
author: "eric"
header-style: text
hidden: false
tags:
  - transformers
  - llama
  - LLM
  - AIGC
---

## 样例 classfication

~~~ python
import transformers import pipeline

## step1
pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert")

## step2
pipe("i like you")

## {'label': 'LABEL_0', 'score': 0.503549337387085}
~~~

### 源码详解

#### step1 

- 根据入参模型 获取模型实例  tokenizer实例 框架 config
    - base::infer_framework_load_model

#### step2
    
- 根据入参 进行分类
    - 执行逻辑
        - 模型是text_classification的，执行text_classification::TextGenerationPipeline::__call__ 
            - base::Pipeline::__call__ 
            - base::Pipeline::run_single（输入为单个文本)
                - text_classification::TextGenerationPipeline::preprocess  构造tokenizer
                - base::Pipeline::forward
                    - base::Pipeline::_ensure_tensor_on_device  inputs->ids
                    - text_classification::TextGenerationPipeline::_forward
                        - modeling_distilbert::DistilBertForSequenceClassification::forward 模型前向函数
                     - base::Pipeline::_ensure_tensor_on_device  返回 ModelOutput([('logits', tensor([[ 0.0100, -0.0041]]))])
                - text_classification::TextGenerationPipeline::postprocess
                    - text_classification::softmax/sigmod
                        - softmax
                            - np.max
                            - np.exp
                        - sigmod
                            - np.exp

                - 返回固定格式 outputs
                
        - outputs

####  详细步骤解析

>  输入  i like  you 

##### pt 编码

> {'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 'input_ids': tensor([[  2,  51,  54, 794, 810, 792,  67, 790, 799,   3]])}

##### forward 

###### _ensure_tensor_on_device 

> {'input_ids': tensor([[  2,  51,  54, 794, 810, 792,  67, 790, 799,   3]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

###### _forward

-  模型 distilbert forward 处理

> 获取第一个元素  tensor([[  2,  51,  54, 794, 810, 792,  67, 790, 799,   3]])

- 预分类层
> tensor([[-0.0883, -0.0462, -0.1825, -0.0073,  0.1329,  0.0698, -0.0872, -0.1736,
          0.1522, -0.0580, -0.1996,  0.0074,  0.1291, -0.0009,  0.2073, -0.1366,
         -0.1998, -0.0499,  0.0625,  0.0365,  0.1001, -0.0114, -0.1269, -0.0806,
         -0.0868, -0.0433, -0.0317,  0.0308,  0.0423, -0.0451,  0.0257, -0.0718]])

- ReLU（Rectified Linear Unit）
> 算子是一种常用的激活函数，它主要用于引入非线性，使得神经网络能够学习和表达更复杂的函数。ReLU 算子将输入的负值变为零，正值保持不变。
> nn.ReLU 
> tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.1329, 0.0698, 0.0000, 0.0000, 0.1522,
         0.0000, 0.0000, 0.0074, 0.1291, 0.0000, 0.2073, 0.0000, 0.0000, 0.0000,
         0.0625, 0.0365, 0.1001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0308, 0.0423, 0.0000, 0.0257, 0.0000]])

- Dropout层
>  self.dropout = nn.Dropout(config.seq_classif_dropout)
> ##定义了Dropout 层进行正则化 防止过拟合 工作原理： 训练过程中随机的以一定概率将输入张量一部分元素设置为0
> config.seq_classif_dropout 为 0.1，则表示有 10% 的输入单元会被随机设置为零
> tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.1329, 0.0698, 0.0000, 0.0000, 0.1522,
         0.0000, 0.0000, 0.0074, 0.1291, 0.0000, 0.2073, 0.0000, 0.0000, 0.0000,
         0.0625, 0.0365, 0.1001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0308, 0.0423, 0.0000, 0.0257, 0.0000]])

- 分类层
>  self.classifier = nn.Linear(config.dim, config.num_labels)
>  tensor([[ 0.0101, -0.0041]])


-  返回结果 tensor([[ 0.0101, -0.0041]])

###### _ensure_tensor_on_device 

-  ModelOutput([('logits', tensor([[ 0.0101, -0.0041]]))])

##### postprocess

###### softmax / sigmod 

- softmax 函数是机器学习和神经网络中常用的激活函数之一，通常用于多分类任务的输出层。它将一个包含任意实数的向量转换为一个概率分布向量，其中每个元素的值在 0 到 1 之间，并且所有元素的和为 1。
- numpy.exp 函数是 NumPy 库中的一个函数，用于计算输入数组中每个元素的指数值
- numpy.argmax 函数用于在指定轴上返回最大值的索引。这个函数对于需要找出数组中最大元素的位置（索引）时非常有用，尤其是在机器学习和数据处理任务中。
- Sigmoid 函数是一个常用的激活函数，通常用于神经网络的隐藏层或输出层。它将任意实数值映射到一个介于 0 和 1 之间的值

> tensor([[ 0.0101, -0.0041]]) --softmax--> [0.50356376 0.49643627]
> [0.50356376 0.49643627] -- argmax --> 0 # 获取索引0

## text_genration

~~~
import transformers import pipeline

text_generator = pipeline(task="text-generation", model="sshleifer/tiny-ctrl", framework="pt")

# Using `do_sample=False` to force deterministic output
outputs = text_generator("This is a test", do_sample=False)


~~~

### 源码详解

-  输入 This is a test

-  tokenize {'input_ids': tensor([[  93,    8,    5, 2549]]), 'token_type_ids': tensor([[0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1]])}

- model genrate tensor([[    93,      8,      5,   2549, 215827, 215827, 115731, 115731, 115731,
         210812, 210812, 114378, 114378,  41539,  41539,  64579, 195172, 195172,
          66462,  66462]])

- 通过 PT 的张量调整 tensor([[[    93,      8,      5,   2549, 215827, 215827, 115731, 115731,
          115731, 210812, 210812, 114378, 114378,  41539,  41539,  64579,
          195172, 195172,  66462,  66462]]])

-  [{'generated_text': 'This is a test ☃ ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy oscope. oscope. FiliFili@@'}]

#### generate

-  generate/utils.py generate
-  Generation Modes 生成文本的模式
    * Greedy Search (贪婪搜索): 简单但可能缺乏多样性
    * Beam Search (束搜索): 平衡性好但计算复杂
    * Top-k Sampling (前k个采样):  增加多样性但需控制 k 值 
    * Top-p (Nucleus) Sampling (核采样):多样性和质量兼顾
    * Temperature (温度调节): 灵活调节生成多样性 



