namespace Backend
{
    internal class Program
    {
        static void Main(string[] args)
        {
            InputLayer input = new InputLayer(3);

            // Hmm, why can't the first hidden layer have any number of units other than 10?
            DenseLayer hidden = new DenseLayer(10, Activation.Sigmoid, WeightInitialisation.Random, BiasInitialisation.Random, input);

            DenseLayer hidden2 = new DenseLayer(20, Activation.Tanh, WeightInitialisation.Random, BiasInitialisation.Random, hidden);
            DenseLayer hidden3 = new DenseLayer(20, Activation.Tanh, WeightInitialisation.Random, BiasInitialisation.Random, hidden2);

            // Hmm, why can't the last hidden layer have any number of units other than 10?
            DenseLayer hidden4 = new DenseLayer(10, Activation.ReLU, WeightInitialisation.Random, BiasInitialisation.Random, hidden3);

            DenseLayer output = new DenseLayer(1, Activation.Sigmoid, WeightInitialisation.Random, BiasInitialisation.Random, hidden4);

            LinearModel model = new LinearModel();
            model.AddLayer(input);
            model.AddLayer(hidden);
            model.AddLayer(output);
            model.Compile(CostFunction.MSE);

            model.Train("./dataset.txt", epochs: 500, batchSize: 64);
        }
    }
}