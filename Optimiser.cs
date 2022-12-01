using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public static class Optimiser
    {
        public static void Fit(LinearModel model, string filename)
        {
            // Collecting dataset which the model is trained upon.
            Dataset data = new Dataset(filename, model.GetInputSize(), model.GetOutputSize());

            int batchSize = model.GetBatchSize();
            CostFunction cost = model.GetCostFunction();
            int epochs = model.GetEpochs();
            float learningRate = model.GetLearningRate();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Collects input-output sample of specified size in matrix form.
                Tuple<float[,], float[,]> sample = data.GetData(batchSize);

                Tuple<float[,], List<float[,]>> forwardPropagation = model.ForwardPropagate(sample.Item1);
                float[,] modelOutput = forwardPropagation.Item1;
                List<float[,]> weightedLayerOutputs = forwardPropagation.Item2;

                Stack<DenseLayer> stack = model.GetStack();

                float[] costGrads;

                // Backpropagates errors through the network and adjusts weights and biases accordingly.
                while (stack.IsEmpty() == false)
                {
                    DenseLayer layer = stack.Pop();
                }
            }

        }

        //public static void Fit(ComplexModel model, string filename)
        //{
        //
        //}

        // Generic wrapper function for the cost function implementations.
        public static float ComputeCost(float[] prediction, float[] expected, CostFunction cost)
        {
            if (prediction.Length != expected.Length)
            {
                throw new ArgumentException("Lengths of prediction and expected vectors must match");
            }
            switch (cost)
            {
                case CostFunction.MSE:
                    return Function.MeanSquaredError(expected, prediction);
                case CostFunction.MAE:
                    return Function.MeanAbsoluteError(expected, prediction);
                default:
                    throw new ArgumentException("Cost function must be one of the defined enumeration types");
            }
        }

        // Computes a row vector of cost function values for two matrices containing prediction and expected vectors, then returns the average.
        // Prediction vectors are produced by the model during the forward pass, expected vectors are taken from the dataset.
        // Polymorphism to work on batched output/expectation vectors.
        public static float ComputeCost(float[,] prediction, float[,] expected, CostFunction cost)
        {
            if (prediction.GetLength(0) != expected.GetLength(0) || prediction.GetLength(1) != expected.GetLength(1))
            {
                throw new ArgumentException("Dimensions of predicted vectors and expected vectors must match");
            }
            float[] costs = new float[prediction.GetLength(1)];
            List<float[]> predictionVectors = Function.SplitMatrixIntoVectors(prediction);
            List<float[]> expectedVectors = Function.SplitMatrixIntoVectors(expected);
            for (int i = 0; i < predictionVectors.Count; i++)
            {
                costs[i] = ComputeCost(predictionVectors[i], expectedVectors[i], cost);
            }
            return costs.Sum() / costs.Length;
        }

        // Generic wrapper for computing cost gradients with respect to final layer's activation (prediction).
        public static float[] ComputeCostGrads(float[] prediction, float[] expected, CostFunction cost)
        {
            if (prediction.Length != expected.Length)
            {
                throw new ArgumentException("Lengths of prediction and expected vectors must match");
            }
            switch (cost)
            {
                case CostFunction.MSE:
                    return Function.MeanSquaredErrorDerivative(expected, prediction);
                case CostFunction.MAE:
                    return Function.MeanAbsoluteErrorDerivative(expected, prediction);
                default:
                    throw new ArgumentException("Cost function must be one of the defined enumeration types");
            }
        }

        // Polymorphism to work on batched output/expectation vectors.
        // The nth column of the output matrix is the cost function gradients vector of the nth columns of the prediction and expected vectors matrices respectively.
        public static float[,] ComputeCostGrads(float[,] prediction, float[,] expected, CostFunction cost)
        {
            if (prediction.GetLength(0) != expected.GetLength(0) || prediction.GetLength(1) != expected.GetLength(1))
            {
                throw new ArgumentException("Dimensions of predicted vectors and expected vectors must match");
            }
            float[,] costGrads = new float[prediction.GetLength(0), prediction.GetLength(1)];
            List<float[]> predictionVectors = Function.SplitMatrixIntoVectors(prediction);
            List<float[]> expectedVectors = Function.SplitMatrixIntoVectors(expected);
            float[] costGrad;
            for (int i = 0; i < predictionVectors.Count; i++)
            {
                switch (cost)
                {
                    case CostFunction.MSE:
                        costGrad = Function.MeanSquaredErrorDerivative(expectedVectors[i], predictionVectors[i]);
                        break;
                    case CostFunction.MAE:
                        costGrad = Function.MeanAbsoluteErrorDerivative(expectedVectors[i], predictionVectors[i]);
                        break;
                    default:
                        throw new ArgumentException("Cost function must be one of the defined enumeration types");
                }
                for (int j = 0; j < costGrad.Length; j++)
                {
                    costGrads[j, i] = costGrad[j];
                }
            }
            return costGrads;
        }

        public static float[,] ComputeFinalLayerError(CostFunction cost, Activation finalLayerActivation, float[,] finalLayerOutput, float[,] expected)
        {
            float[,] nablaAC = ComputeCostGrads(finalLayerOutput, expected, cost);
            float[,] sigmaPrimeWeightedOutput = finalLayerOutput;
            switch (finalLayerActivation)
            {
                case Activation.ReLU:
                    Function.Vectorise(sigmaPrimeWeightedOutput, Function.ReLUDerivative);
                    break;
                case Activation.Sigmoid:
                    Function.Vectorise(sigmaPrimeWeightedOutput, Function.SigmoidDerivative);
                    break;
                case Activation.Tanh:
                    Function.Vectorise(sigmaPrimeWeightedOutput, Function.TanhDerivative);
                    break;
                case Activation.SiLU:
                    Function.Vectorise(sigmaPrimeWeightedOutput, Function.SiLUDerivative);
                    break;
                case Activation.None:
                    Function.Vectorise(sigmaPrimeWeightedOutput, Function.NoneDerivative);
                    break;
                default:
                    throw new ArgumentException("Activation must be one of the specified enumeration values");
            }
            return Function.HadamardProduct(nablaAC, sigmaPrimeWeightedOutput);
        }

        public static float[,] ComputeLayerError(DenseLayer layer)
        {

        }
    }
}
