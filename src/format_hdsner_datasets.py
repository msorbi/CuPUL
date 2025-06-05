#!/usr/bin/env python3
import argparse
import os
import shutil

def copy_and_modify_files(input_dir, output_dir, singleclass):
    for dataset in os.listdir(input_dir):
        dataset_path = os.path.join(input_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        output_dataset_path = os.path.join(output_dir, dataset)
        os.makedirs(output_dataset_path, exist_ok=True)

        class_names = []
        # Determine class names for types.txt
        if not singleclass:
            # For MULTICLASS, gather all single classes in the dataset
            for class_dir in os.listdir(dataset_path):
                if os.path.isdir(os.path.join(dataset_path, class_dir)) and class_dir != "MULTICLASS":
                    class_names.append(class_dir)
            class_names.sort(reverse=True) # Ensure consistent order (PERS then LOC)

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

            output_class_path = os.path.join(output_dataset_path, class_type)
            os.makedirs(output_class_path, exist_ok=True)

            for filename in os.listdir(class_path):
                source_file_path = os.path.join(class_path, filename)
                destination_filename = filename

                if filename == "val.txt":
                    destination_filename = "valid.txt"

                destination_file_path = os.path.join(output_class_path, destination_filename)

                if os.path.isfile(source_file_path):
                    if filename == "train.txt":
                        with open(source_file_path, 'r') as infile, \
                             open(destination_file_path, 'w') as outfile:
                            for line in infile:
                                parts = line.strip().split('\t')
                                if len(parts) >= 2:
                                    second_column = parts[1]
                                    class_index = 0  # Default to 0

                                    if '-' in second_column:
                                        extracted_class = second_column.split('-')[-1]
                                        if singleclass:
                                            # If singleclass, the class index is always 0 for the single class
                                            class_index = 0
                                        else:
                                            # For MULTICLASS, find the index of the extracted class
                                            try:
                                                class_index = class_names.index(extracted_class)
                                            except ValueError:
                                                pass # If not found, class_index remains 0

                                    parts.append(str(class_index))
                                    outfile.write('\t'.join(parts) + '\n')
                                else:
                                    outfile.write(line) # Write original line if not enough columns
                    else:
                        shutil.copy2(source_file_path, destination_file_path)

            # Add types.txt
            types_txt_path = os.path.join(output_class_path, "types.txt")
            with open(types_txt_path, 'w') as f:
                if singleclass:
                    f.write(f"{class_type}\n")
                else:
                    for class_name in class_names:
                        f.write(f"{class_name}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Adapt datasets to CuPUL")
    parser.add_argument('--input-dir', type=str, default="../hdsner-utils/data/distant/ner_medieval_multilingual/FR/", help='Path to the datasets directory.')
    parser.add_argument('--output-dir', type=str, default="../data/hdsner/disant", help='Path to the destination directory.')
    parser.add_argument('--singleclass', action='store_true', help='If set, copy only single class directories; otherwise, copy MULTICLASS.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    copy_and_modify_files(args.input_dir, args.output_dir, args.singleclass)

if __name__ == "__main__":
    main()
