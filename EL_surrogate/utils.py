import jax.numpy as jnp
import numpy as np
import ml_collections

def get_dataset(config: ml_collections.ConfigDict):
    # Load dataset
    numpy_array = np.genfromtxt("ELPSD_dataset_train_V2.csv", delimiter=',', skip_header=1)
    dataset = jnp.array(numpy_array)

    # Check expected shape
    expected_cols = config.input_dim + config.output_dim
    if dataset.shape[1] != expected_cols:
        raise ValueError(
            f"Dataset shape mismatch! Expected {expected_cols} columns (inputs + outputs), "
            f"but got {dataset.shape[1]} columns."
        )

    # Perform dataset split if enabled in config
    if config.use_train_test_split or config.use_train_val_test_split:
        num_samples = dataset.shape[0]

        # Shuffle dataset for randomness
        np.random.seed(42)  # Ensure reproducibility
        permuted_indices = np.random.permutation(num_samples)
        dataset = dataset[permuted_indices]

        if config.use_train_val_test_split:
            train_idx = int(0.7 * num_samples)
            test_idx = int(0.9 * num_samples)  # 70% train + 20% test = 90%

            train_data = dataset[:train_idx, :]
            test_data = dataset[train_idx:test_idx, :]
            val_data = dataset[test_idx:, :]  # Last 10% for validation

            # Save test & validation data to CSV files
            np.savetxt("test_data.csv", np.array(test_data), delimiter=",", header=",".join([f"Feature_{i}" for i in range(expected_cols)]), comments="")
            np.savetxt("validation_data.csv", np.array(val_data), delimiter=",", header=",".join([f"Feature_{i}" for i in range(expected_cols)]), comments="")

            print(f"Dataset split: {train_data.shape[0]} training, {test_data.shape[0]} test, {val_data.shape[0]} validation samples")
            print("Test data saved to test_data.csv")
            print("Validation data saved to validation_data.csv")

            return train_data  # Return only training data

        else:
            # **Default Split: 80% Train, 20% Test**
            split_idx = int(0.8 * num_samples)
            train_data = dataset[:split_idx, :]
            test_data = dataset[split_idx:, :]

            # Save test data to CSV file
            np.savetxt("test_data.csv", np.array(test_data), delimiter=",", header=",".join([f"Feature_{i}" for i in range(expected_cols)]), comments="")

            print(f"Dataset split: {train_data.shape[0]} training, {test_data.shape[0]} test samples")
            print("Test data saved to test_data.csv")

            return train_data  # Return only training data

    # If no split is required, return the full dataset
    else:
        return dataset
