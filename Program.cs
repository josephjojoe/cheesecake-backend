namespace Backend
{
    internal class Program
    {
        static void Main(string[] args)
        {
            InputLayer inputLayer = new InputLayer(4);
            DenseLayer denseBranch1 = new DenseLayer(10, Activation.Tanh, WeightInitialisation.Xavier, inputLayer);
            DenseLayer denseBranch2 = new DenseLayer(5, Activation.Sigmoid, WeightInitialisation.Xavier, inputLayer);
            DenseLayer outputLayer = new DenseLayer(10, Activation.Tanh,
                                               WeightInitialisation.Xavier,
                                               new List<Layer>() { denseBranch1, denseBranch2 },
                                               MergeType.Concatenate);


            float[,] input = new float[4, 3] { { 1, 4, 6 },
                                               { 1, 4, 5 },
                                               { 3, 6, 7 },
                                               { 3, 5, 6 } };
            input = inputLayer.ForwardPass(input);

            float[,] inputBranch1 = denseBranch1.ForwardPass(input);
            float[,] inputBranch2 = denseBranch2.ForwardPass(input);

            Console.WriteLine($"{inputBranch1.GetLength(0)}, {inputBranch1.GetLength(1)}");
            Console.WriteLine($"{inputBranch2.GetLength(0)}, {inputBranch2.GetLength(1)}");

            float[,] outputraaa = Function.Concatenate(new List<float[,]>() { inputBranch1, inputBranch2});
            Console.WriteLine($"{outputraaa.GetLength(0)}, {outputraaa.GetLength(1)}");

            Console.ReadLine();

            input = outputLayer.ForwardPass(new List<float[,]>() { inputBranch1, inputBranch2 });

            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    Console.Write($"{input[i, j]} ");
                }
                Console.Write("\n");
            }
            Console.ReadLine();
        }
    }
}