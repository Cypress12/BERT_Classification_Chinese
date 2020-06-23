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