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

                // List to hold costs - displays average at the end of each epoch.
                List<float> costs = new List<float>();

                while (sample != null)
                {
                    if (sample.Item1.Length == 0)
                    {
                        break;
                    }

                    List<float[,]> errors = new List<float[,]>();
                    float[,] modelOutput = model.ForwardPropagate(sample.Item1);

                    List<Layer> layers = model.GetLayers();

                    float costValue = ComputeCost(modelOutput, sample.Item2, model.GetCostFunction());
                    costs.Add(costValue);

                    errors.Add(ComputeFinalLayerError(model.GetCostFunction(), layers[layers.Count - 1].GetActivation(),
                        layers[layers.Count - 1].GetWeightedOutput(), layers[layers.Count - 1].GetActivationOutput(), sample.Item2));

                    // Recursive error computation after final layer error is calculated.
                    for (int i = layers.Count - 2; i > 0; i--)
                    {
                        errors.Add(ComputeLayerError(errors[layers.Count - 2 - i], ((DenseLayer)layers[i + 1]).GetWeights(), layers[i].GetActivation(), layers[i].GetWeightedOutput()));
                    }

                    // Modifies model parameters and collects new sample for next loop iteration.
                    model.TrainingStep(errors);
                    sample = data.GetData(batchSize);
                }

                // Displays cost.
                Console.WriteLine($"Epoch {epoch} cost: {costs.Sum() / costs.Count}");
                // Resets index pointer so that data can be used in next epoch for training.
                data.ResetDatabase();
            }
        }

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

        public static float[,] ComputeFinalLayerError(CostFunction cost, Activation finalLayerActivationFunction, float[,] finalLayerWeightedOutput,
                                                      float[,] finalLayerActivationOutput,float[,] expected)
        {
            float[,] nablaAC = ComputeCostGrads(finalLayerActivationOutput, expected, cost);
            float[,] sigmaPrimeWeightedOutput = finalLayerWeightedOutput;
            switch (finalLayerActivationFunction)
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

        public static float[,] ComputeLayerError(float[,] followingLayerError, float[,] followingLayerWeights, Activation layerActivation, float[,] layerWeightedOutput)
        {
            float[,] errorComponentOne = Function.Multiply(Function.Transpose(followingLayerWeights), followingLayerError);
            float[,] errorComponentTwo = layerWeightedOutput;
            switch (layerActivation)
            {
                case Activation.ReLU:
                    Function.Vectorise(errorComponentTwo, Function.ReLUDerivative);
                    break;
                case Activation.Sigmoid:
                    Function.Vectorise(errorComponentTwo, Function.SigmoidDerivative);
                    break;
                case Activation.Tanh:
                    Function.Vectorise(errorComponentTwo, Function.TanhDerivative);
                    break;
                case Activation.SiLU:
                    Function.Vectorise(errorComponentTwo, Function.SiLUDerivative);
                    break;
                case Activation.None:
                    Function.Vectorise(errorComponentTwo, Function.NoneDerivative);
                    break;
                default:
                    throw new ArgumentException("Activation must be one of the specified enumeration values");
            }
            return Function.HadamardProduct(errorComponentOne, errorComponentTwo);
        }
    }
}
