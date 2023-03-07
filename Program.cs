namespace Backend
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //InputLayer input = new InputLayer(3); 
            //DenseLayer hidden = new DenseLayer(30, Activation.Sigmoid, WeightInitialisation.Random, BiasInitialisation.Random, input);
            //DenseLayer hidden2 = new DenseLayer(20, Activation.Tanh, WeightInitialisation.Random, BiasInitialisation.Random, hidden);
            //DenseLayer hidden3 = new DenseLayer(20, Activation.Tanh, WeightInitialisation.Random, BiasInitialisation.Random, hidden2);
            //DenseLayer hidden4 = new DenseLayer(15, Activation.ReLU, WeightInitialisation.Random, BiasInitialisation.Random, hidden3);
            //DenseLayer output = new DenseLayer(1, Activation.Sigmoid, WeightInitialisation.Random, BiasInitialisation.Random, hidden4);

            //LinearModel model = new LinearModel();
            //model.AddLayer(input);
            //model.AddLayer(hidden);
            //model.AddLayer(hidden2);
            //model.AddLayer(hidden3);
            //model.AddLayer(hidden4);
            //model.AddLayer(output);
            //model.Compile(CostFunction.MSE);

            //model.Train("./dataset.txt", epochs: 500, batchSize: 64);

            InputLayer input = new InputLayer(1);

            DenseLayer dense = new DenseLayer(2, Activation.Sigmoid, WeightInitialisation.Random, BiasInitialisation.Random, input);
            DenseLayer dense2 = new DenseLayer(3, Activation.Sigmoid, WeightInitialisation.Random, BiasInitialisation.Random, dense);

            DenseLayer dense_stack1_1 = new DenseLayer(4, Activation.Sigmoid,
                                                       WeightInitialisation.Random, BiasInitialisation.Random, dense2);
            DenseLayer dense_stack1_2 = new DenseLayer(5, Activation.Sigmoid,
                                                       WeightInitialisation.Random, BiasInitialisation.Random, dense_stack1_1);

            DenseLayer dense_stack2_1 = new DenseLayer(6, Activation.Sigmoid,
                                                       WeightInitialisation.Random, BiasInitialisation.Random, dense2);
            DenseLayer dense_stack2_2 = new DenseLayer(7, Activation.Sigmoid,
                                                       WeightInitialisation.Random, BiasInitialisation.Random, dense_stack2_1);
            DenseLayer dense_stack2_3 = new DenseLayer(8, Activation.Sigmoid,
                                                       WeightInitialisation.Random, BiasInitialisation.Random, dense_stack2_2);

            DenseLayer dense_merge_1 = new DenseLayer(9, Activation.Sigmoid,
                                                      WeightInitialisation.Random, BiasInitialisation.Random,
                                                      previousLayers: new List<Layer>() { dense_stack1_2, dense_stack2_3 }, MergeType.Concatenate);

            ComplexModel model = new ComplexModel();
            model.AddInputLayer(input);
            model.AddLayer(input, dense);
            model.AddLayer(dense, dense2);

            model.AddLayer(dense2, dense_stack1_1);
            model.AddLayer(dense_stack1_1, dense_stack1_2);

            model.AddLayer(dense2, dense_stack2_1);
            model.AddLayer(dense_stack2_1, dense_stack2_2);
            model.AddLayer(dense_stack2_2, dense_stack2_3);

            model.AddLayer(dense_stack1_2, dense_merge_1);
            model.AddLayer(dense_stack2_3, dense_merge_1);

            float[] inputdata = new float[1] { 6.7f };
            float[] output = model.ForwardPropagate(inputdata);
            foreach (float a in output)
            {
                Console.WriteLine(a);
            }
        }
    }
}