#!/usr/bin/env python3
import argparse
import os
import shutil

def copy_and_modify_files(input_dir, output_prefix, singleclass):
    for dataset in os.listdir(input_dir):
        dataset_path = os.path.join(input_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        output_dataset_path = '-'.join([output_prefix, dataset])

        class_names = []
        # Determine class names for types.txt
        if not singleclass:
            # For MULTICLASS, gather all single classes in the dataset
            for class_dir in os.listdir(dataset_path):
                if os.path.isdir(os.path.join(dataset_path, class_dir)) and class_dir != "MULTICLASS":
                    class_names.append(class_dir)
            class_names.sort(reverse=True) # Ensure consistent order (PERS then LOC)
            label_map = {'O': 0}
            num_labels = 1
            for entity_type in class_names:
                label_map['B-'+entity_type] = num_labels
                label_map['I-'+entity_type] = num_labels
                num_labels += 1

        for class_type in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_type)
            if not os.path.isdir(class_path):
                continue

            if singleclass:
                if class_type == "MULTICLASS":
                    continue
            else: # not singleclass
                if class_type != "MULTICLASS":
                    continue

            output_class_path = '-'.join([output_dataset_path, class_type])
            os.makedirs(output_class_path, exist_ok=True)

            for filename in os.listdir(class_path):
                if filename.endswith("stats.json"):
                    continue
                source_file_path = os.path.join(class_path, filename)
                destination_filename = filename

                if filename == "val.txt":
                    destination_filename = "valid.txt"

                destination_file_path = os.path.join(output_class_path, destination_filename)

                if os.path.isfile(source_file_path):
                    if filename in ["train.txt", "val.txt", "test.txt"]:
                        with open(source_file_path, 'r') as infile, \
                             open(destination_file_path, 'w') as outfile:
                            for line in infile:
                                line = line.replace(' ','_').replace('\t', ' ')
                                parts = line.strip().split(' ')
                                outfile.write(line)
                            outfile.write("\n")
                    else:
                        shutil.copy2(source_file_path, destination_file_path)

            # Add types.txt
            types_txt_path = os.path.join(output_class_path, "types.txt")
            with open(types_txt_path, 'w') as f:
                if singleclass:
                    f.write(f"{class_type}")
                else:
                    f.write("\n".join(class_names))

def parse_args():
    parser = argparse.ArgumentParser(description="Adapt datasets to CuPUL")
    parser.add_argument('--input-dir', type=str, default="../hdsner-utils/data/distant/ner_medieval_multilingual/FR/", help='Path to the datasets directory.')
    parser.add_argument('--output-prefix', type=str, default="../data/hdsner-distant", help='Path to the destination directory.')
    parser.add_argument('--singleclass', action='store_true', help='If set, copy only single class directories; otherwise, copy MULTICLASS.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    copy_and_modify_files(args.input_dir, args.output_prefix, args.singleclass)

if __name__ == "__main__":
    main()
