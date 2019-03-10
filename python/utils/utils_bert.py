import os
import numpy as np
import pandas as pd


def compute_offset_no_spaces(text, offset):
	count = 0
	for pos in range(offset):
		if text[pos] != " ": count +=1
	return count

def count_chars_no_special(text):
	count = 0
	special_char_list = ["#"]
	for pos in range(len(text)):
		if text[pos] not in special_char_list: count +=1
	return count

def count_length_no_special(text):
	count = 0
	special_char_list = ["#", " "]
	for pos in range(len(text)):
		if text[pos] not in special_char_list: count +=1
	return count

def run_bert(data):
	'''
	Runs a forward propagation of BERT on input text, extracting contextual word embeddings
	Input: data, a pandas DataFrame containing the information in one of the GAP files

	Output: emb, a pandas DataFrame containing contextual embeddings for the words A, B and Pronoun. Each embedding is a numpy array of shape (768)
	columns: "emb_A": the embedding for word A
	         "emb_B": the embedding for word B
	         "emb_P": the embedding for the pronoun
	         "label": the answer to the coreference problem: "A", "B" or "NEITHER"
	'''
    # From the current file, take the text only, and write it in a file which will be passed to BERT
	text = data["Text"]
	text.to_csv("../data/input.txt", index = False, header = False)

    # The script extract_features.py runs forward propagation through BERT, and writes the output in the file output.jsonl
    # I'm lazy, so I'm only saving the output of the last layer. Feel free to change --layers = -1 to save the output of other layers.
	os.system("python3 ./utils/extract_features.py \
	  --input_file=../data/input.txt \
	  --output_file=../data/output.jsonl \
	  --vocab_file=./uncased_L-12_H-768_A-12/vocab.txt \
	  --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json \
	  --init_checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt \
	  --layers=-1 \
	  --max_seq_length=256 \
	  --batch_size=8")

	bert_output = pd.read_json("../data/output.jsonl", lines = True)

	os.system("rm ../data/output.jsonl")
	os.system("rm ../data/input.txt")

	index = data.index
	columns = ["emb_A", "emb_B", "emb_P", "label"]
	emb = pd.DataFrame(index = index, columns = columns)
	emb.index.name = "ID"

	for i in range(len(data)): # For each line in the data file
		# get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
		P = data.loc[i,"Pronoun"].lower()
		A = data.loc[i,"A"].lower()
		B = data.loc[i,"B"].lower()

		# For each word, find the offset not counting spaces. This is necessary for comparison with the output of BERT
		P_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"Pronoun-offset"])
		A_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"A-offset"])
		B_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"B-offset"])
		# Figure out the length of A, B, not counting spaces or special characters
		A_length = count_length_no_special(A)
		B_length = count_length_no_special(B)

		# Initialize embeddings with zeros
		emb_A = np.zeros(768)
		emb_B = np.zeros(768)
		emb_P = np.zeros(768)

		# Initialize counts
		count_chars = 0
		cnt_A, cnt_B, cnt_P = 0, 0, 0

		features = pd.DataFrame(bert_output.loc[i,"features"]) # Get the BERT embeddings for the current line in the data file
		for j in range(2,len(features)):  # Iterate over the BERT tokens for the current line; we skip over the first 2 tokens, which don't correspond to words
			token = features.loc[j,"token"]

			# See if the character count until the current token matches the offset of any of the 3 target words
			if count_chars  == P_offset: 
				# print(token)
				emb_P += np.array(features.loc[j,"layers"][0]['values'])
				cnt_P += 1
			if count_chars in range(A_offset, A_offset + A_length): 
				# print(token)
				emb_A += np.array(features.loc[j,"layers"][0]['values'])
				cnt_A +=1
			if count_chars in range(B_offset, B_offset + B_length): 
				# print(token)
				emb_B += np.array(features.loc[j,"layers"][0]['values'])
				cnt_B +=1								
			# Update the character count
			count_chars += count_length_no_special(token)
		# Taking the average between tokens in the span of A or B, so divide the current value by the count	
		emb_A /= cnt_A
		emb_B /= cnt_B

		# Work out the label of the current piece of text
		label = "Neither"
		if (data.loc[i,"A-coref"] == True):
			label = "A"
		if (data.loc[i,"B-coref"] == True):
			label = "B"

		# Put everything together in emb
		emb.iloc[i] = [emb_A, emb_B, emb_P, label]

	return emb
