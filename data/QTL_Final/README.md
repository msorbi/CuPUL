# QTL_Final Dataset

This directory contains the final dataset and model predictions used for QTL trait extraction experiments.

## Files Description

- **`train.txt` / `valid.txt` / `test.txt`**  
  The training, validation, and test splits of the dataset in BIO format.

- **`types.txt`**  
  A list of all possible entity types.  
  **Note:** Although the `gene` type appears in `types.txt` and `valid.txt`, it is **not used** in this project and is treated as type `O` (outside any entity).

- **`pred_test_high_recall.txt`**  
  Predictions from a model optimized for **high recall**. This model favors capturing more relevant entities, even if that introduces more false positives.

- **`pred_test_balance.txt`**  
  Predictions from a model optimized for a **balanced trade-off** between precision and recall, aiming for more stable overall performance.
