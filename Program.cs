namespace Backend
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
        }

        public class DenseLayer
        {
            private float[,]? _weights;    // Weights are null until initialised with model compilation, as input data shape must be known.
            private bool _weightsInitialised = false;    // Flag to check if weights are initialised yet (must be done before forward pass).
            private float[] _bias;
            private int _units;
            private Activation _activation;
            private int _inputSize;
            private WeightInitialisation _weightInitialisation;    // Weight initialisation type should be known at time of layer construction.

            public DenseLayer(int units, Activation activation, WeightInitialisation weightInitialisation)
            {
                _units = units;
                _bias = new float[units];
                _activation = activation;
                _weightInitialisation = weightInitialisation;
            }

            public int GetOutputSize()    // Output size of one layer is the input size of the next (barring multi-branch layer topologies).
            {
                return _units;
            }

            public void InitialiseWeights(int inputSize)
            {
                if (_weightsInitialised == true)
                {
                    // Throw some error - weights should not be initialised twice for a layer.
                }
                _inputSize = inputSize;
                _weights = new float[_units, _inputSize];    // Is this the right way around for the weights matrix? Make sure to test.
                switch (_weightInitialisation)
                {
                    case WeightInitialisation.Ones:    // Anonymous functions that initialise weights.
                        Vectorise(_weights, x => 1);
                        break;
                    case WeightInitialisation.Zeroes:
                        Vectorise(_weights, x => 0);
                        break;
                    case WeightInitialisation.Random:
                        Random rngRandom = new Random();
                        Vectorise(_weights, x => (float)rngRandom.NextDouble());
                        break;
                    case WeightInitialisation.Xavier:
                        Random rngXavier = new Random();
                        float lower = (float)-(1.0 / Math.Sqrt(_inputSize)), upper = (float)(1.0 / Math.Sqrt(_inputSize));
                        Vectorise(_weights, x => lower + ((float)rngXavier.NextDouble() * (upper - lower)));
                        break;
                }
                _weightsInitialised = true;    // Set weights initialised to true after this function is called.
            }

            public float[] forwardPass(float[] inputs)
            {
                if (inputs.Length != _inputSize)
                {
                    // Throw some error.
                }
                if (_weightsInitialised == false)
                {
                    // Throw another error.
                }

                // Multiply weights matrix by input vector and add activation - implement then overload some Multiply(float[], float[]) function.
                // Add support for multiplying single- and multi-dimensional arrays.
                throw new NotImplementedException("Implement me!");
            }
        }


        public enum Activation { ReLU, Sigmoid, Tanh, Swish, Softmax }
        public enum WeightInitialisation { Zeroes, Ones, Random, Xavier };

        // Delegate to allow for vectorisation of functions - application of the function to every element of an array.
        public delegate float VectorisationDelegate(float input);

        // Vectorise function for single-dimensional arrays.
        public static void Vectorise(float[] vector, VectorisationDelegate function)
        {
            for (int index = 0; index < vector.Length; index++)
            {
                vector[index] = function(vector[index]);
            }
        }

        // Overloading Vectorize to work on multi-dimensional arrays.
        public static void Vectorise(float[,] array, VectorisationDelegate function)
        {
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    array[i, j] = function(array[i, j]);
                }
            }
        }

        public static void Multiply()
        {

        }
    }
}