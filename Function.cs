﻿namespace Backend
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


        // Definition as per http://neuralnetworksanddeeplearning.com/chap2.html for a single training example.
        public static float MeanSquaredError(float[] expected, float[] output)
        {
            if (expected.Length != output.Length)
            {
                throw new ArgumentException("Errors cannot be computed for vectors of different lengths");
            }
            float sum = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                sum += (float)Math.Pow((expected[i] - output[i]), 2);
            }
            return sum / 2;
        }

        // Computes gradient (delta)aC of the MSE cost function with respect to the activations (the output vector).
        public static float[] MeanSquaredErrorDerivative(float[] expected, float[] output)
        {
            if (expected.Length != output.Length)
            {
                throw new ArgumentException("Errors cannot be computed for vectors of different lengths");
            }
            Function.Vectorise(expected, x => x * -1);
            return Function.Add(output, expected);
        }

        public static float MeanAbsoluteError(float[] expected, float[] output)
        {
            if (expected.Length != output.Length)
            {
                throw new ArgumentException("Errors cannot be computed for vectors of different lengths");
            }
            float sum = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                sum += Math.Abs(expected[i] - output[i]);
            }
            return sum / expected.Length;
        }

        // Implementation as per https://stats.stackexchange.com/questions/312737
        public static float[] MeanAbsoluteErrorDerivative(float[] expected, float[] output)
        {
            if (expected.Length != output.Length)
            {
                throw new ArgumentException("Errors cannot be computed for vectors of different lengths");
            }
            float[] grads = new float[expected.Length];
            for (int i = 0; i < expected.Length; i++)
            {
                if (output[i] > expected[i])
                {
                    grads[i] = 1;
                }
                else if (output[i] < expected[i])
                {
                    grads[i] = -1;
                }
                else
                {
                    // Arbitrary choice of gradient made when y_pred = y_true.
                    grads[i] = 0;
                }
            }
            return grads;
        }

        public static List<float[]> SplitMatrixIntoVectors(float[,] matrix)
        {
            List<float[]> result = new List<float[]>();
            for (int i = 0; i < matrix.GetLength(1); i++)
            {
                float[] vector = new float[matrix.GetLength(0)];
                for (int j = 0; j < matrix.GetLength(0); j++)
                {
                    vector[j] = matrix[j, i];
                }
                result.Add(vector);
            }
            return result;
        }

        // Returns a vector which is the average of the column vectors of the matrix.
        public static float[] AverageMatrix(float[,] matrix)
        {
            float[] result = new float[matrix.GetLength(0)];
            List<float[]> columns = SplitMatrixIntoVectors(matrix);
            for (int i = 0; i < result.Length; i++)
            {
                float temp = 0;
                for (int j = 0; j < columns.Count; j++)
                {
                    temp += columns[j][i];
                }
                result[i] = temp / columns.Count;
            }
            return result;
        }

        // Takes an (Mx1) and a (1xN) vector and gives an (MxN) matrix.
        // In the context of my project, the column vector will be the activation from the previous layer.
        // The row vector will be the error of the current layer.
        public static float[,] OuterProduct(float[] columnVector, float[] rowVector)
        {
            float[,] result = new float[columnVector.Length, rowVector.Length];
            for (int i = 0; i < result.GetLength(0); i++)
            {
                for (int j = 0; j < result.GetLength(1); j++)
                {
                    result[i, j] = columnVector[i] * rowVector[j];
                }
            }
            return result;
        }

        public static float[,] Transpose(float[,] matrix)
        {
            float[,] transpose = new float[matrix.GetLength(1), matrix.GetLength(0)];
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    transpose[j, i] = matrix[i, j];
                }
            }
            return transpose;
        }

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
                throw new ArgumentException("Dimensions of arrays to be multiplied must match");
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
                throw new ArgumentException("Dimensions of arrays to be multiplied must match");
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
                throw new ArgumentException("Dimensions of arrays to be added must match");
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
                throw new ArgumentException("Dimensions of arrays to be added must match");
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
        public static float[,] RowConcatenate(List<float[,]> matrices)
        {
            if (matrices.Count == 1)
            {
                return matrices[0];
            }
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

        // Returns matrix with input vectors as columns.
        public static float[,] ConcatenateVectorsIntoMatrix(List<float[]> vectors)
        {
            List<int> vectorLengths = new List<int>();
            foreach (float[] vector in vectors)
            {
                vectorLengths.Add(vector.Length);
            }
            if (vectorLengths.Distinct().Count() > 1)
            {
                throw new Exception("Vectors must all have the same length");
            }

            float[,] matrix = new float[vectors[0].Length, vectors.Count];
            for (int i = 0; i < vectors.Count; i++)
            {
                for (int j = 0; j < vectors[0].Length; j++)
                {
                    matrix[j, i] = vectors[i][j];
                }
            }
            return matrix;
        }

        public static float[,] HadamardProduct(float[,] matrix1, float[,] matrix2)
        {
            if (!(matrix1.GetLength(0) == matrix2.GetLength(0) && matrix1.GetLength(1) == matrix2.GetLength(1)))
            {
                throw new ArgumentException("Dimensions of arrays to be multiplied must match");
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
                throw new ArgumentException("Dimensions of arrays to be multiplied must match");
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