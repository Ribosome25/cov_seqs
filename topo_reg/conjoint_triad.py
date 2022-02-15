"""
From https://github.com/MahaAmin/Conjoint-Triad/blob/c26c2e542c582dc34e7dcd4c5296f3841b76e357/conjoint_triad.py


"""


from decimal import *
import csv
from topo_reg.io import check_path_exists, read_df, save_df
from topo_reg.args import ConjointTriadArgs
from Bio import SeqIO
import pandas as pd

# Generating vector space (7*7*7)
def VS(rang):
    V = []
    for i in range(1,rang):
        for j in range(1,rang):
            for k in range(1,rang):
                tmp = "VS"+str(k) + str(j) + str(i)
                V.append(tmp)
    return V


# calculating conjoint triad for input sequence
def frequency(seq:str) -> list:
    # 注 seq supports slicing. Str should be ok. 
    frequency = []
    for i in range(0, (len(seq) - 3)):
        subSeq = seq[i:i+3]
        tmp = "VS"
        for j in range(0,3):
            if((subSeq[j] == 'A') or (subSeq[j] == 'G') or (subSeq[j] == 'V')):
                tmp += "1"
            elif((subSeq[j] == 'I') or (subSeq[j] == 'L') or (subSeq[j] == 'F') or (subSeq[j] == 'P')):
                tmp += "2"
            elif((subSeq[j] == 'Y') or (subSeq[j] == 'M') or (subSeq[j] == 'T') or (subSeq[j] == 'S')):
                tmp += "3"
            elif((subSeq[j] == 'H') or (subSeq[j] == 'N') or (subSeq[j] == 'Q') or (subSeq[j] == 'W')):
                tmp += "4"
            elif((subSeq[j] == 'R') or (subSeq[j] == 'K')):
                tmp += "5"
            elif((subSeq[j] == 'D') or (subSeq[j] == 'E')):
                tmp += "6"
            elif((subSeq[j] == 'C')):
                tmp += "7"
        frequency.append(tmp)
    return frequency


# Creating frequency_dictionary, and calaculate frequency for eaech conjoint triad
def freq_dict(V: list, freq: list) -> dict:
    # 注：非常慢的方法
    frequency_dictionary = {}
    for i in range(0, len(V)):
        key = V[i]
        frequency_dictionary[key] = 0

    for i in range(0, len(freq)):
        frequency_dictionary[freq[i]] = frequency_dictionary[freq[i]]+1

    return frequency_dictionary


# Export the output to .csv file
def output_to_csv(seq_ID, frequency_dict):

    # Each row in csv file [ seqID, frequencies ]
    data = [seq_ID]
    for key, value in frequency_dict.items():
        data.append(value)

    with open('conjoint_triad.csv', 'a') as csvfile:
        conjointTriad = csv.writer(csvfile)
        conjointTriad.writerow(data)


# Reading sequences from fasta file.
def fasta_input():
    print("Enter path of .fasta file : ", end='')
    path = input()
    sequences = []
    seq_IDs = []
    for record in SeqIO.parse(path, "fasta"):
        sequences.append(record.seq)
        seq_IDs.append(record.id)
    return sequences, seq_IDs

def conjoint_triad(sequences: list, seq_IDs: list = None):
    if seq_IDs is None:
        seq_IDs = ["I{}".format(x) for x in range(len(sequences))]

    v = VS(8)

    #writing vector space as header in the csv file
    header = ["ID"]
    for i in range(0, len(v)):
        header.append(v[i])

    all_results = {}
    # calculating conjoint_triad for each sequence
    for i in range(0, len(sequences)):
        fi = frequency(sequences[i])
        freqDict = freq_dict(v, fi)
        all_results[seq_IDs[i]] = freqDict
        # output_to_csv(seq_IDs[i], freqDict)
    # print('Data was exported to "conjoint_triad.csv."')
    return pd.DataFrame.from_dict(all_results).T

def embed_conjoint_triad(args: ConjointTriadArgs):
    df = read_df(args.df_path)
    seqs = df['Seqs']
    idx = df.index
    ct = conjoint_triad(seqs, idx)
    check_path_exists(args.save_path)
    save_df(ct, args.save_path)


if __name__ == '__main__':
    conjoint_triad()
