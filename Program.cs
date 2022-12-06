namespace Backend
{
    internal class Program
    {
        static void Main(string[] args)
        {
            InputLayer input = new InputLayer(3);
            DenseLayer hidden = new DenseLayer(10, Activation.Sigmoid, WeightInitialisation.Xavier, BiasInitialisation.Xavier, input);
            DenseLayer output = new DenseLayer(1, Activation.Sigmoid, WeightInitialisation.Xavier, BiasInitialisation.Xavier, hidden);

            LinearModel model = new LinearModel();
            model.AddLayer(input);
            model.AddLayer(hidden);
            model.AddLayer(output);
            model.Compile(CostFunction.MSE);

            model.Train("./dataset.txt", batchSize: 64);
        }
    }
}