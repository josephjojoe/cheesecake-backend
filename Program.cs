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
            private bool _weightsInitialised = false;    // Flag to check if weights are initialised yet (must be done before forward pass), prevents double initialisation.
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
                _weights = new float[_units, _inputSize];    // Is this the right way around for the weights matrix? Make sure to test!
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
        // Only allows functions that map floats to floats - error checking is therefore not required in Vectorise and associated overloads.
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

        public static float[,] Multiply(float[,] array1, float[,] array2)
        {
            if (array1.GetLength(1) != array2.GetLength(0))
            {
                // Throw some error - dimensions must match.
                Console.WriteLine("BAD BAD BAD");
            }
            float[,] output = new float[array1.GetLength(0), array2.GetLength(1)];
            float temp = 0;    // Holds temporary values from matrix multiplication calculation.
            for (int i = 0; i < array1.GetLength(0); i++)
            {
                for (int j = 0; j < array2.GetLength(1); j++)
                {
                    temp = 0;
                    for (int k = 0; k < array1.GetLength(1); k++)
                    {
                        temp += array1[i, k] * array2[k, j];
                    }
                    output[i, j] = temp;
                }
            }
            return output;
        }

        // Overloading Multiply to support matrix-vector multiplication.
        public static float[] Multiply(float[,] matrix, float[] vector)
        {
            if (matrix.GetLength(1) != vector.Length)
            {
                // Throw some error - dimensions must match.
            }
            float[] output = new float[vector.Length];
            float temp;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                temp = 0;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    temp += matrix[i, j] * vector[j];
                }
                output[i] = temp;
            }
            return output;
        }

        public static float[] Add(float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                // Throw some error - dimensions must match for element-wise addition.
            }
            float[] output = new float[vector1.Length];
            for (int i = 0; i < vector1.Length; i++)
            {
                output[i] = vector1[i] + vector2[i];
            }
            return output;
        }
    }
}