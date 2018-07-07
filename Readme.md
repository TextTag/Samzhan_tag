### Dependency:

tensorflow > 1.0.0

python == 2.7.12



### Preprocess:

参数在代码最后一行：

```python
Data_extraction('raw_data', 'MySheet', './usr.dict', './stop_words.txt', 'all', 'test_raw', True)
```

| 字段             | 意义                           |
| -------------- | ---------------------------- |
| raw_data       | 训练原始文件夹（可包含多个训练文件）           |
| MySheet        | 数据所在Sheet，这个导师给的名字老变，最好统一下   |
| usr.dict       | jieba分词需要load的dict           |
| stop_words.txt | 停用词集                         |
| all            | 预处理后数据所在文件夹名，前缀是‘data_’      |
| test_raw       | ‘看点’ 源文件的文件夹，用于最后测试          |
| True           | True生成train和dev， False生成test |

```
python Preprocess.py
```

这个写的有点糙，主要是前期需求，数据格式，不太稳定，不过逻辑简单，很好改，最终确认再改改。这步完成后会生成可训的训练文件或者测试文件。



### Train:

建立model文件夹

```
mkdir yourmodeldir
```

生成config文件（超参数表）

```
python train_test.py --weight-path yourmodeldir
```

训练

```
CUDA_VISIBLE_DEVICES=1 python train_test.py --weight-path yourmodeldir --load-config
```



### Test:

```
CUDA_VISIBLE_DEVICES=1 python train_test.py --weight-path yourmodeldir --load-config --train-test test > all_res.txt
```

res.txt即为预测结果

我服务器中文件目录

```
.
├── all_model
│   ├── checkpoint
│   ├── classifier.weights.data-00000-of-00001
│   ├── classifier.weights.index
│   ├── classifier.weights.meta
│   ├── config
│   └── run.log
├── all_res.txt
├── Config.py
├── Config.pyc
├── data_all
│   ├── all_res.txt
│   ├── dev
│   ├── dict
│   ├── test
│   ├── tmp_label
│   └── train
├── data_entertain
│   ├── dev
│   ├── dict
│   └── train
├── data_finance
│   ├── dev
│   ├── dict
│   ├── test
│   ├── tmp_label
│   └── train
├── data_helpers.py
├── data_helpers.pyc
├── dict
├── entertain_model
│   ├── checkpoint
│   ├── classifier.weights.data-00000-of-00001
│   ├── classifier.weights.index
│   ├── classifier.weights.meta
│   ├── config
│   └── run.log
├── finance_model
│   ├── checkpoint
│   ├── classifier.weights.data-00000-of-00001
│   ├── classifier.weights.index
│   ├── classifier.weights.meta
│   ├── config
│   └── run.log
├── Model_father.py
├── Model_father.pyc
├── Preproess.py
├── raw_data
│   ├── toutiao_content_image_2.xlsx
│   └── toutiao_content_image.xlsx
├── res.txt
├── SeqLabel_model.py
├── SeqLabel_model.pyc
├── stop_words.txt
├── test_raw
│   └── kd_new_0703-2.xls
├── TfUtils.py
├── TfUtils.pyc
├── Train_father.py
├── Train_father.pyc
├── train_test.py
├── usr.dict
├── util.py
└── util.pyc
```

