namespace Backend
{
    // Static class for all the mathematics functions I will be using.
    public static class Function
    {
        public static float ReLU(float n)
        {
            if (n > 0) { return n; }
            else { return 0; }
        }

        public static float ReLUDerivative(float n)
        {
            if (n > 0) { return 1; }
            else { return 0; }
        }

        public static float Sigmoid(float n)
        {
            return (float)(1 / (1 + Math.Exp(n * -1)));
        }

        public static float SigmoidDerivative(float n)
        {
            return Sigmoid(n) * (1 - Sigmoid(n));
        }

        public static float SiLU(float n)
        {
            return (float)(n / (1 + Math.Exp(n * -1)));
        }

        public static float SiLUDerivative(float n)
        {
            return (float)(Math.Exp(n) * (n + Math.Exp(n) + 1) / Math.Pow(Math.Exp(n) + 1, 2));
        }

        public static float Tanh(float n)
        {
            return (float)(Math.Exp(n) - Math.Exp(n * -1)) / (float)(Math.Exp(n) + Math.Exp(n * -1));
        }

        public static float TanhDerivative(float n)
        {
            return (float)(1 - Math.Pow(Tanh(n), 2));
        }

        public static float None(float n)
        {
            return n;
        }

        public static float NoneDerivative(float n)
        {
            return 1;
        }

        public readonly static Dictionary<Func<float, float>, Func<float, float>> ActivationDictionary = new Dictionary<Func<float, float>, Func<float, float>>
        {{ ReLU, ReLUDerivative }, { Sigmoid, SigmoidDerivative }, { Tanh, TanhDerivative }, { SiLU, SiLUDerivative }, { None, NoneDerivative }};


        // Create actual definitions for these.
        public static float MeanSquaredError(float output, float expectedOutput)
        {
            return 1;
        }

        public static float MeanSquaredErrorDerivative(float output, float expectedOutput)
        {
            return 1;
        }

        public static float MeanAbsoluteError(float output, float expectedOutput)
        {
            return 1;
        }

        public static float MeanAbsoluteErrorDerivative(float output, float expectedOutput)
        {
            return 1;
        }

        // Func<float, float, float> means that we are dealing with functions that take in two float arguments and output a float.
        public readonly static Dictionary<Func<float, float, float>, Func<float, float, float>> CostDictionary = new Dictionary<Func<float, float, float>, Func<float, float, float>>
        {{ MeanSquaredError, MeanSquaredErrorDerivative }, { MeanAbsoluteError, MeanAbsoluteErrorDerivative }};

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

        // Predominantly used for two or more layers merging into one layer with the additional condition
        // that the input is a matrix of vectors, such as in stochastic mini-batch gradient descent.
        // Given a matrix with dimensions (BxA) and another matrix with dimensions (CxA), will output a
        // matrix that concatenates them with dimensions ((B+C)xA). Extended to an arbitrary number of matrices.
        public static float[,] Concatenate(List<float[,]> matrices)
        {
            int rows = 0;
            int columns = matrices[0].GetLength(1);
            foreach (float[,] matrix in matrices)
            {
                rows += matrix.GetLength(0);
            }
            float[,] output = new float[rows, columns];
            int offset = 0;
            foreach (float[,] matrix in matrices)
            {
                for (int i = 0; i < matrix.GetLength(0); i++)
                {
                    for (int j = 0; j < matrix.GetLength(1); j++)
                    {
                        output[i + offset, j] = matrix[i, j];
                    }
                }
                offset += matrix.GetLength(0);
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

        // Creates a matrix with the same column vector repeated - used for the bias term when processing batched inputs.
        public static float[,] GetVectorAsMatrix(float[] vector, int columns)
        {
            float[,] output = new float[vector.Length, columns];
            for (int i = 0; i < vector.Length; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    output[i, j] = vector[i];
                }
            }
            return output;
        }
    }
}