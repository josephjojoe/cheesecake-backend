namespace Backend
{
    internal class Program
    {
        static void Main(string[] args)
        {

        }

        public class AddLayer : Layer
        {
            private Layer[] _inputLayers;

            public AddLayer(Layer[] inputLayers)
            {
                // Need check on if all input Layers have the same output size, else throw error.
                _inputLayers = inputLayers;
            }

            public float[] ForwardPass()
            {
                throw new NotImplementedException();
            }
        }

        public class ConcatenateLayer : Layer
        {
            private Layer[] _inputLayers;

            public ConcatenateLayer(Layer[] inputLayers)
            {
                _inputLayers = inputLayers;
            }

            public float[] ForwardPass()
            {
                throw new NotImplementedException();
            }
        }

        public class DenseLayer : Layer
        {
            private float[,]? _weights;    // Weights are null until initialised with model compilation, as input data shape must be known.
            private bool _weightsInitialised = false;    // Flag to check if weights are initialised yet (must be done before forward pass), prevents double initialisation.
            private float[] _bias;
            private int _units;
            private Activation _activation;
            private int _inputSize;
            private WeightInitialisation _weightInitialisation;    // Weight initialisation type should be known at time of layer construction.
            private Layer _previousLayer;

            public DenseLayer(int units, Activation activation, WeightInitialisation weightInitialisation, Layer previousLayer)
            {
                _units = units;
                _bias = new float[units];
                _activation = activation;
                _weightInitialisation = weightInitialisation;
                _previousLayer = previousLayer;
            }

            public int GetOutputSize()    // Output size of one layer is the input size of the next (barring multi-branch layer topologies).
            {
                return _units;
            }

            public Activation GetActivation()
            {
                return _activation;
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

            public override float[] ForwardPass(float[] inputs)
            {
                if (inputs.Length != _inputSize)
                {
                    // Throw some error.
                }
                if (_weightsInitialised == false)
                {
                    // Throw another error.
                }

                float[] output = new float[_inputSize];
                output = Multiply(_weights, inputs);

                switch (GetActivation())
                {
                    case Activation.ReLU:
                        Vectorise(output, ReLU);
                        return output;
                    case Activation.Sigmoid:
                        Vectorise(output, Sigmoid);
                        return output;
                    case Activation.SiLU:
                        Vectorise(output, SiLU);
                        return output;
                    case Activation.Tanh:
                        return output;
                    case Activation.None:
                        break;    // Returns output outside of switch-case so that all code paths return a value.
                }
                return output;
            }

            public void ModifyWeights(float[,] modification)
            {
                if (_weightsInitialised == false)
                {
                    // Throw some error - weights must be initialised.
                }
                Add(_weights, modification);
            }

            public float[,] GetWeights()
            {
                if (_weightsInitialised == false)
                {
                    // Throw some error - weights must be initialised.
                }
                return _weights;
            }

            public float[] GetWeightsDimension()
            {
                if (_weightsInitialised == false)
                {
                    // Throw some error - weights must be initialised.
                }
                return new float[2] { _weights.GetLength(0), _weights.GetLength(1) };
            }

            public void ModifyBias(float[] modification)
            {
                Add(_bias, modification);
            }

            public float[] GetBias()
            {
                return _bias;
            }

            public int GetUnits()
            {
                return _units;
            }
        }

        // An input layer has no weights, biases, or activations.
        // Its sole purpose is to propagate an input through the network during training and inference.
        // It acts as a springboard from which the rest of the neural network is built.
        public class InputLayer : Layer
        {
            // Input size is also referred to as 'shape' in other libraries - it allows other layer input/output shapes to be determined.
            // Batch size is only used for stochastic mini-batch gradient descent in this project; if not supplied to the
            // constructor, we assume a batch size of 1 which means that no shuffling and buffer system is used.
            private int _inputSize;
            private int _batchSize;

            public InputLayer(int inputSize, int batchSize = 1)
            {
                _inputSize = inputSize;
                _batchSize = batchSize;
            }

            public override float[] ForwardPass(float[] input)
            {
                return input;
            }
        }

        public abstract class Layer
        {
            public abstract float[] ForwardPass(float[] input);
        }

        public static float ReLU(float n)
        {
            if (n < 0) { return 0; }
            else { return n; }
        }

        public static float Sigmoid(float n)
        {
            return (float)(1 / (1 + Math.Exp(n * -1)));
        }

        public static float SiLU(float n)
        {
            return (float)(1 / (1 + Math.Exp(n * -1)) * n);
        }

        public static float Tanh(float n)
        {
            return (float)(Math.Exp(n) - Math.Exp(n * -1)) / (float)(Math.Exp(n) + Math.Exp(n * -1));
        }

        public enum Activation { ReLU, Sigmoid, Tanh, SiLU, None }
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
            float[] output = new float[matrix.GetLength(0)];
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

        // Overloading Add for element-wise addition of matrices.
        public static float[,] Add(float[,] matrix1, float[,] matrix2)
        {
            if (!(matrix1.GetLength(0) == matrix2.GetLength(0) && matrix1.GetLength(1) == matrix2.GetLength(1)))
            {
                // Throw some error - dimensions must match for Hadamard product.
            }
            float[,] output = new float[matrix1.GetLength(0), matrix1.GetLength(1)];
            for (int i = 0; i < matrix1.GetLength(0); i++)
            {
                for (int j = 0; j < matrix1.GetLength(1); j++)
                {
                    output[i, j] = matrix1[i, j] + matrix2[i, j];
                }
            }
            return output;
        }

        public static float[,] HadamardProduct(float[,] matrix1, float[,] matrix2)
        {
            if (!(matrix1.GetLength(0) == matrix2.GetLength(0) && matrix1.GetLength(1) == matrix2.GetLength(1)))
            {
                // Throw some error - dimensions must match for Hadamard product.
            }
            float[,] output = new float[matrix1.GetLength(0), matrix1.GetLength(1)];
            for (int i = 0; i < matrix1.GetLength(0); i++)
            {
                for (int j = 0; j < matrix1.GetLength(1); j++)
                {
                    output[i, j] = matrix1[i, j] * matrix2[i, j];
                }
            }
            return output;
        }

        // Overloading HadamardProduct to work on single-dimensional vectors.
        public static float[] HadamardProduct(float[] vector1, float[] vector2)
        {
            if (!(vector1.Length == vector2.Length))
            {
                // Throw some error - dimensions must match.
            }
            float[] output = new float[vector1.Length];
            for (int i = 0; i < vector1.Length; i++)
            {
                output[i] = vector1[i] * vector2[i];
            }
            return output;
        }
    }
}