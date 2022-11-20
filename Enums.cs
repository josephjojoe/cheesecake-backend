namespace Backend
{
    public enum Activation { ReLU, Sigmoid, Tanh, SiLU, None }

    public enum WeightInitialisation { Zeroes, Ones, Random, Xavier };

    // For merging multiple layer branches to one layer.
    // Add merges branch outputs by adding them (requires all branch outputs to have the same dimension).
    // Concatenate merges by concatenating to a larger vector.
    public enum MergeType { Add, Concatenate };

    public enum BiasInitialisation { Zeroes, Ones, Random, Xavier };

    public enum CostFunction { MSE, MAE };
}

