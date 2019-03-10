import os


text = data["Text"]
text.to_csv("../data/input.txt", index = False, header = False)

os.system("python3 extract_features.py \
	  --input_file=input.txt \
	  --output_file=output.jsonl \
	  --vocab_file=./uncased_L-12_H-768_A-12/vocab.txt \
	  --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json \
	  --init_checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt \
	  --layers=-1 \
	  --max_seq_length=256 \
	  --batch_size=8")