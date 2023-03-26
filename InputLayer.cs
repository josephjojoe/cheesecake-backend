using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    // An input layer has no weights, biases, or activations.
    // Its sole purpose is to propagate an input through the network during training and inference.
    // It acts as a springboard from which the rest of the neural network is built, and an entry point for the optimiser.
    public class InputLayer : Layer
    {
        // Input size is also referred to as 'shape' in other libraries - it allows other layer input/output shapes to be determined.
        private int _inputSize;
        private float[,]? _weightedOutput;
        private float[,]? _activationOutput;

        public InputLayer(int inputSize)
        {
            _inputSize = inputSize;
        }

        public override float[,] GetWeightedOutput()
        {
            return _weightedOutput;
        }

        public override float[,] GetActivationOutput()
        {
            return _activationOutput;
        }

        public override float[] ForwardPass(float[] input)
        {
            return input;
        }

        // Override for stochastic mini-batch gradient descent.
        public override float[,] ForwardPass(float[,] inputs)
        {
            _weightedOutput = inputs;
            _activationOutput = inputs;
            return inputs;
        }

        // No weights and biases, so weighted output is the same.
        public override float[] WeightedOutput(float[] input)
        {
            return input;
        }

        // Override for stochastic mini-batch gradient descent.
        public override float[,] WeightedOutput(float[,] inputs)
        {
            return inputs;
        }

        public override int GetOutputSize()
        {
            return _inputSize;
        }

        public override Activation GetActivation()
        {
            return Activation.None;
        }
    }
}
