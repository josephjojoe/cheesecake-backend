using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
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

        public override int GetOutputSize()
        {
            return _inputSize;
        }
    }
}
