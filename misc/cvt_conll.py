import jsonlines
import os
import argparse
import glob
from tqdm import tqdm


def convert_file(input_name, sup_output_name, query_output_name):
    if not os.path.exists(sup_output_name):
        os.makedirs(sup_output_name)
    #okay
    if not os.path.exists(query_output_name):
        os.makedirs(query_output_name)
    reader = jsonlines.open(input_name)
    for ct, dicts in enumerate(reader):
        supdict = dicts["support"]
        supwords = supdict["word"]
        suplabels = supdict["label"]
        querydict = dicts["query"]
        querywords = querydict["word"]
        querylabels = querydict["label"]
        str1 = ''
        for i in range(len(supwords)):
            for j in range(len(supwords[i])):
                str1 = str1 + supwords[i][j] + '\t' + suplabels[i][j] + '\n'
            str1 += '\n'
        out_file = open(sup_output_name + '/' + str(ct) + '.txt', 'w', encoding='utf-8')
        out_file.write(str1)
        out_file.close()
        str2 = ''
        for i in range(len(querywords)):
            for j in range(len(querywords[i])):
                str2 = str2 + querywords[i][j] + '\t' + querylabels[i][j] + '\n'
            str2 += '\n'
        out_file = open(query_output_name + '/' + str(ct) + '.txt', 'w', encoding='utf-8')
        out_file.write(str2)
        out_file.close()
    

if __name__ == '__main__':
    output_base_dir = 'data/few-nerd'
    support_file_prefix = 'support_'
    query_file_prefix = 'query_'
    print(os.getcwd())    
    # ensure the folders are already created beforehand
    
    all_input_files = glob.glob('**/*.jsonl', recursive=True)

    for file in tqdm(all_input_files):
        input_base_name = os.path.basename(file).split('.')[0] # just take the base name
        if 'test' in file:
            target_split_text = 'inter' if 'inter' in file else 'intra'
            sup_output_name = os.path.join(output_base_dir, target_split_text, support_file_prefix + input_base_name)
            query_output_name = os.path.join(output_base_dir, target_split_text, query_file_prefix + input_base_name)
            convert_file(file, sup_output_name, query_output_name)
        
        