namespace Backend
{
    internal class Program
    {
        static void Main(string[] args)
        {
            InputLayer inputLayer = new InputLayer(4);
            DenseLayer denseBranch1 = new DenseLayer(5, Activation.Tanh, WeightInitialisation.Xavier, inputLayer);
            DenseLayer denseBranch2 = new DenseLayer(10, Activation.Sigmoid, WeightInitialisation.Xavier, inputLayer);
            DenseLayer output = new DenseLayer(10, Activation.Tanh,
                                               WeightInitialisation.Xavier,
                                               new List<Layer>() { denseBranch1, denseBranch2 },
                                               MergeType.Concatenate);


            float[] input = new float[4] { 1, 4, 6, 3 };
            input = inputLayer.ForwardPass(input);

            float[] inputBranch1 = denseBranch1.ForwardPass(input);
            float[] inputBranch2 = denseBranch2.ForwardPass(input);
            input = output.ForwardPass(new List<float[]>() { inputBranch1, inputBranch2 });
            for (int i = 0; i < input.Length; i++)
            {
                Console.WriteLine(input[i]);
            }
            Console.ReadLine();
        }
    }
}