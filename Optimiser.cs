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

            List<Layer> layers = model.GetLayers();
            layers.Reverse();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Collects input-output sample of specified size in matrix form.
                Tuple<float[,], float[,]> sample = data.GetData(batchSize);
                Console.WriteLine(sample.Item1.GetLength(0));
                Console.WriteLine(sample.Item1.GetLength(1));
                Console.ReadLine();

                while (sample != null)
                {
                    List<float[,]> errors = new List<float[,]>();
                    Console.WriteLine("You made it?");
                    Console.ReadLine();
                    float[,] modelOutput = model.ForwardPropagate(sample.Item1);
                    modelOutput = model.ForwardPropagate(sample.Item1);
                    Console.WriteLine("You made it!");
                    Console.ReadLine();

                    float costValue = ComputeCost(modelOutput, sample.Item2, model.GetCostFunction());


                    errors.Add(ComputeFinalLayerError(model.GetCostFunction(), layers[0].GetActivation(), layers[0].GetWeightedOutput(),
                        layers[0].GetActivationOutput(), sample.Item2));


                    // 'layers.Count - 1' excludes the final layer (input layer) in these for loops.
                    for (int i = 1; i < layers.Count - 1; i++)
                    {
                        errors.Add(ComputeLayerError(errors[i - 1], ((DenseLayer)layers[i - 1]).GetWeights(), layers[i].GetActivation(), layers[i].GetWeightedOutput()));
                    }

                    Console.WriteLine($"Amount of errors: {errors.Count}");
                    Console.ReadLine();

                    for (int i = 0; i < layers.Count - 1; i++)
                    {
                        // Previous layer is layers[i + 1] as the list of layers has been reversed.
                        ModifyWeightsAndBiases(errors[i], (DenseLayer)layers[i], layers[i + 1], learningRate);
                    }

                    // sample = data.GetData(batchSize);
                }
                // Resets index pointer so that data can be used in next epoch for training.
                data.ResetDatabase();
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

        public static void ModifyWeightsAndBiases(float[,] error, DenseLayer layer, Layer previousLayer, float eta)
        {
            ModifyBiases(error, layer, eta);
            // ModifyWeights(error, layer, previousLayer, eta);
        }

        public static void ModifyBiases(float[,] error, DenseLayer layer, float eta)
        {
            float[] averagedError = Function.AverageMatrix(error);
            Function.Vectorise(averagedError, x => eta * x);
            // Adds error * eta (learning rate) to the biases to improve them.
            layer.ModifyBias(averagedError);
        }

        public static void ModifyWeights(float[,] error, DenseLayer layer, Layer previousLayer, float eta)
        {
            float[] averagedActivation = Function.AverageMatrix(previousLayer.GetActivationOutput());
            float[] averagedError = Function.AverageMatrix(error);
            float[,] modification = Function.Transpose(Function.OuterProduct(averagedActivation, averagedError));

            // Multiplies by eta (learning rate) then adds to weights to improve them.
            Function.Vectorise(modification, x => eta * x);
            layer.ModifyWeights(modification);
        }
    }
}
