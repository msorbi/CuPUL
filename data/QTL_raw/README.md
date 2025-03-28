# QTL_raw

This directory contains the raw data, dictionaries, and distant supervision resources used for QTL-related entity annotation tasks.

## data/QTL

This folder includes the main dataset splits and their distant supervision variants:

- **`train.txt` / `valid.txt` / `test.txt`**  
  The standard training, validation, and test sets.

- **`train.ALL.txt`**  
  Distant supervision output where **both `gene` and `trait` entities** are annotated.

- **`train.Gene.txt`**  
  Only **`gene` entities** are annotated using distant supervision.

- **`train.Trait.txt`**  
  Only **`trait` entities** are annotated using distant supervision.

- **`test_distant_labeling.txt`**  
  Distant labeling version of the test set.

## dictionaries

This directory contains the dictionaries used for distant supervision.

### `QTL/`

- **`Trait.txt`**  
  A **high-quality, manually curated** trait dictionary, written with reference to ontology terms.

- **`Gene.txt`**  
  A gene dictionary used for tagging gene mentions.

- **`Trait.txt.ontology`**  
  A **large but noisy** dictionary **directly extracted from ontology** resources. Useful for recall-focused tasks.

## dict_match.py

This script performs **distant labeling** using the provided dictionaries. It matches dictionary terms in raw text and produces weakly supervised annotations for training.
