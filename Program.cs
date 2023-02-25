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

            DirectedAcyclicGraph graph = new DirectedAcyclicGraph();

            graph.AddInputNode(input);
            graph.AddConnection(input, dense);
            graph.AddConnection(dense, dense2);

            graph.TopologicalSort();
            graph.TopologicalSort();
            List<Layer> traverseOrder = graph.GetTopologicalSort();

            Console.WriteLine($"Number of nodes in traversal: {traverseOrder.Count}");
            Console.WriteLine($"Number of units in each layer to test traversal order:");
            for (int i = 0; i < traverseOrder.Count; i++)
            {
                Console.WriteLine(traverseOrder[i].GetOutputSize());
            }
            Console.ReadLine();

            graph.AddConnection(dense2, dense_stack1_1);
            graph.AddConnection(dense_stack1_1, dense_stack1_2);

            graph.AddConnection(dense2, dense_stack2_1);
            graph.AddConnection(dense_stack2_1, dense_stack2_2);
            graph.AddConnection(dense_stack2_2, dense_stack2_3);

            graph.AddConnection(dense_stack1_2, dense_merge_1);
            graph.AddConnection(dense_stack2_3, dense_merge_1);

            graph.TopologicalSort();
            traverseOrder = graph.GetTopologicalSort();

            Console.WriteLine($"Number of nodes in traversal: {traverseOrder.Count}");
            Console.WriteLine($"Number of units in each layer to test traversal order:");
            for (int i = 0; i < traverseOrder.Count; i++)
            {
                Console.WriteLine(traverseOrder[i].GetOutputSize());
            }
            Console.ReadLine();
        }
    }
}