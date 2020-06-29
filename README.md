# BERT_Classification_Chinese

利用BERT模型进行中文情感二分类项目

开发环境：Google colaboratory

### 数据集

本次实验数据在data文件夹，其中yiqing.csv是2020年1月至2月疫情话题微博原始数据集，具体格式如下：

```
   wb_id：微博ID
   wb_username：微博用户名
   wb_userid：微博用户ID
   wb_content：微博内容
   wb_createtime：微博发布时间
   wb_forwardnum：微博转发数
   wb_commentnum：微博评论数
   wb_likenum：微博点赞数
   wb_url：微博链接
```
  
 其余为标注数据集，数据格式为
 
    label,valid
   
 label为单条样本标签（消极0 or 积极1），valid为微博内容
 
 ----
 以下是项目具体步骤：
 
 ### 1、下载预训练模型
 
 略，模型已上传，在chinese_L-12_H-768_A-12文件夹
 
 ### 2、修改Processor
 
 在run_classsifier.py文件中我们可以看到，google对于一些公开数据集已经写了一些Processor，如XnliProcessor,MnliProcessor,MrpcProcessor和ColaProcessor，因此对于自己的Processor依样画葫芦~~复制粘贴~~即可。对应文本二分类任务，get_labels函数写成如下的形式：
 
 ```
 def get_labels(self):
        return ['0', '1']
 ```
 
 修改完成Processor后，需要在在原本main函数的Processor字典里加入修改后的Processor类，即可在运行参数里指定调用该Processor
 
 ```
 processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "weibo": WeiboProcessor
  }
  ```
  
  ### 3、添加评价指标
  
  源代码只有acc和loss的计算，本次实验在metric_fn函数添加了其余模型评价指标，修改后的metric_fn函数如下：
  
  ```
        def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        precision = tf.metrics.precision(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        f1_score = tf.contrib.metrics.f1_score(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        recall = tf.metrics.recall(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        auc = tf.metrics.auc(labels=label_ids, predictions=predictions, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall":recall,
            "eval_F1_score":f1_score,
            "eval_loss": loss,
            "eval_auc":auc,
        }
  ```
  
  ### 4、运行fine-tune
  
  之后就可以直接运行run_classsifier.py进行模型的训练。在运行时需要制定一些参数，本项目的运行参数如下所示：
  
  ```
export BERT_BASE_DIR="./chinese_L-12_H-768_A-12"
export WEIBO_DIR="./data/"

python ./run_classifier.py \
  --task_name=WeiBo \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$WEIBO_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=27 \
  --learning_rate=5e-6 \
  --num_train_epochs=2.0 \
  --max_seq_length=100 \
  --output_dir=./output/
  ```
  
  其中do_train进行训练，do_eval进行验证，do_predict进行预测，如不需要将对应项改为false即可
