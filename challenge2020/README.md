# Physionetchallenge2020
```
├─models                                /模型构建文件
│      inceptiontime.py
│      inceptiontime_v2.py
│      resnet.py
│      resnext.py
│
├─nas                                   / Neural architecture search
│  │  inceptiontime.py                  / nas for inceptiontime
│  │  resnet_v2.py                      / nas for resnet34
│  │
│  ├─files                              / 该文件夹下内容需要替换到autokeras和kerastuner相应目录下
│  │  │  readme.txt                     / 详细说明
│  │  │
│  │  ├─autokeras                       
│  │  │
│  │  └─kerastuner
│  │
│  └─utils                              / 工具类
│
├─preprocess                            / 数据预处理
│      preprocess-challenge2018.py
│
├─raw_data                             
└─utils                                 / 工具类
|       data.py                        
|       data_2.py
|       loss.py
|       tools.py
|
│  main_inceptiontime.py                
│  main_resnet.py                       
│  README.md

...