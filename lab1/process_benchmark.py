import os
import sys

import pandas as pd

def read_file(file):
    with open(file, 'r') as f:
        data = f.readlines()

    data = [x for x in data if x != '\n']
    count_headers = len([x for x in data if x.startswith('#')])
    return data[count_headers-1:]

def get_headers(header_line):
    tokens = header_line.split()
    tokens = tokens[1:]

    if len(tokens) == 2:
        return tokens
    
    if len(tokens) == 3:
        return [tokens[0], tokens[1]+tokens[2]]
    
    if len(tokens) == 4:
        return [tokens[0]+tokens[1], tokens[2]+tokens[3]]
    else:
        raise Exception('Header parsing failed')

def create_table(file):

    lines = read_file(file)
    headers = get_headers(lines[0])
    
    records = []

    for line in lines[1:]:
        cols = line.split()
        records.append({headers[0]: int(cols[0]), headers[1]: float(cols[1])})

    df = pd.DataFrame.from_records(records)
    print(df.to_latex(index=False, bold_rows=True, 
                      float_format="{:.2f}".format))

    return df


if __name__ == "__main__":
    print(sys.argv[1])
    dir_name = sys.argv[1]
    files = os.listdir(dir_name)

    for file in files:
        print(file)
        print('\n')
        create_table(sys.argv[1]+'/'+file)
        print('\n\n')
