import csv

def process_csv(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        headers = next(reader)
        headers.insert(0, 'dataset')
        writer.writerow(headers)
        for row in reader:
            split_values = row[0].split('/', 2)
            if split_values[0] == 'ali':
                dataset_value = split_values[0]
                field1_value = '/'.join(split_values[2:])
            else:
                dataset_value = split_values[0]
                field1_value = '/'.join(split_values[1:])
            new_row = [dataset_value, field1_value] + row[1:]
            writer.writerow(new_row)

for i in range(8):
    input_file = f'/nfs/data/clap/merged-backup/{i}/clap_metas.csv'
    output_file = f'/nfs/data/clap/merged/{i}/clap_metas.csv'
    process_csv(input_file, output_file)