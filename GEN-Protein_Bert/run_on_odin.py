from Bio import SeqIO, motifs
from Bio.Seq import Seq
import pandas as pd
import numpy as np
from Bio.SeqRecord import SeqRecord
import torch
import json
from transformers import AutoTokenizer, AutoModel

def encode_sequences(sequences):
    return [tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]

def embed_sequences(encoded_sequences):
    input_ids = torch.tensor(encoded_sequences)
    with torch.no_grad():
        embeddings = model(input_ids)
    return embeddings

def dimention_reduction(embeddings_def):
    dimention_reduction_result = []
    for i in range(1024):
        dimention_reduction_result.append(np.mean(embeddings_def[:,i].tolist()))
    return dimention_reduction_result

# testing
fasta_pos_test = "testing/human_test_positive.fasta"
with open(fasta_pos_test, "r") as file3:
    sequences_pos_test = list(SeqIO.parse(file3, "fasta"))
sequence_data = []
for seq_record in sequences_pos_test:
    sequence_data.append({"Sequence": str(seq_record.seq),"Label":1})
df_test_pos = pd.DataFrame(sequence_data)
    
fasta_neg_test = "testing/human_test_negative.fasta"
with open(fasta_neg_test, "r") as file4:
    sequences_neg_test = list(SeqIO.parse(file4, "fasta"))    
sequence_data = []
for seq_record in sequences_neg_test:
    sequence_data.append({"Sequence": str(seq_record.seq),"Label":0})
df_test_neg = pd.DataFrame(sequence_data)

testing = pd.concat([df_test_pos, df_test_neg], ignore_index=True)

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert")

for n in range(len(testing['Sequence'])):
    print("test" + str(n))
    if n == 0:
        sequences = testing['Sequence'][n]
        encoded_sequences = encode_sequences(sequences)
        embeddings = embed_sequences(encoded_sequences)
        feature_temp = {
            '0': dimention_reduction(embeddings[1])
        }
        feature_test = pd.DataFrame.from_dict(feature_temp, orient='index')
    else:
        sequences = testing['Sequence'][n]
        encoded_sequences = encode_sequences(sequences)
        embeddings = embed_sequences(encoded_sequences)
        feature_test.loc[n] = dimention_reduction(embeddings[1])
feature_test.to_csv('feature_test.csv', index=False)