using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public class LinearModel : Model
    {
        // List to hold layers -> Transfer to queue with Compile().
        List<Layer> _layers = new List<Layer>();
        // Nullable until Compile() called.
        Queue<Layer>? _queue;
        // Boolean that indicates whether the model is ready to use and train - Compile() is a prerequisite.
        bool _readyToTrain = false;
        int _batchSize;
        CostFunction _costFunction;
        float _learningRate;
        int _epochs;

        public void AddLayer(Layer layer)
        {
            _layers.Add(layer);
        }

        // Uses the errors generated in backpropagation to change the model's weights and biases.
        public void TrainingStep(List<float[,]> errors)
        {
            // Okay this loop isn't executing for some reason.
            // Okay the reason is that errors doesn't have all the errors computed for some reason.
            for (int i = _layers.Count - 1; i > 0; i--)
            {
                ModifyWeightsAndBiases(errors[_layers.Count - 1 - i], (DenseLayer)_layers[i], _layers[i - 1], _learningRate);
            }
            float[,] newWeights = ((DenseLayer)_layers[1]).GetWeights();

            // Pushes trained layer attributes onto queue for future forward passes.
            Requeue();
        }

        // Code to modify the weights and biases based on the errors generated in backpropagation.
        public static void ModifyWeightsAndBiases(float[,] error, DenseLayer layer, Layer previousLayer, float eta)
        {
            float[] averagedError = Function.AverageMatrix(error);

            // For gradient descent, we are multiplying eta by -1 so that we go against the direction of steepest ascent.
            // For gradient ascent, we would omit multiplying eta by -1.
            eta = eta * -1;

            Function.Vectorise(averagedError, x => eta * x);
            // Adds error * eta (learning rate) to the biases to improve them. Eta means that not overly large steps are made.
            layer.ModifyBias(averagedError);

            float[] averagedActivation = Function.AverageMatrix(previousLayer.GetActivationOutput());
            float[,] modification = Function.Transpose(Function.OuterProduct(averagedActivation, averagedError));
            // Multiplies by eta (learning rate) then adds to weights to improve them.
            Function.Vectorise(modification, x => eta * x);
            layer.ModifyWeights(modification);
        }

        // float[] is the output of the model.
        public float[] ForwardPropagate(float[] input)
        {
            if (_readyToTrain == false)
            {
                throw new Exception("Model must be compiled before training");
            }
            else
            {
                for (int i = 0; i < _queue.GetQueueSize(); i++)
                {
                    Layer layer = _queue.Dequeue();
                    input = layer.ForwardPass(input);
                }
                Requeue();
                return input;
            }
        }

        // Polymorphism for batched inputs.
        public override float[,] ForwardPropagate(float[,] input)
        {
            if (_readyToTrain == false)
            {
                throw new Exception("Model must be compiled before training");
            }
            else
            {
                for (int i = 0; i < _queue.GetQueueSize(); i++)
                {
                    Layer layer = _queue.Dequeue();
                    input = layer.ForwardPass(input);
                }
                Requeue();
                return input;
            }
        }

        // Pushes layers back onto Queue for next forward propagation.
        public void Requeue()
        {
            _queue.ResetPointers();
            for (int i = 0; i < _layers.Count; i++)
            {
                _queue.Enqueue(_layers[i]);
            }
        }

        public override void Compile(CostFunction costFunction)
        {
            _costFunction = costFunction;

            if (_layers[0] is not InputLayer)
            {
                throw new ArgumentException("First layer must be Input layer");
            }
            if (_layers.FindAll(s => s is InputLayer).Count > 1)
            {
                throw new ArgumentException("Only one input layer is allowed");
            }

            _queue = new Queue<Layer>(_layers.Count());
            foreach (Layer l in _layers)
            {
                _queue.Enqueue(l);
            }
            _readyToTrain = true;
        }

        public void Train(string filename, int epochs, float learningRate, int batchSize)
        {
            if (_readyToTrain == false)
            {
                throw new Exception("Model must be compiled before training");
            }
            if (batchSize < 1)
            {
                throw new ArgumentException("Batch size cannot be less than 1");
            }

            _epochs = epochs;
            _learningRate = learningRate;
            _batchSize = batchSize;

            // Filename refers to file name of or path to the dataset file.
            Optimiser.Fit(this, filename);
        }

        public int GetBatchSize()
        {
            return _batchSize;
        }

        public CostFunction GetCostFunction()
        {
            return _costFunction;
        }

        public float GetLearningRate()
        {
            return _learningRate;
        }

        public int GetEpochs()
        {
            return _epochs;
        }

        // Next two methods are used by the optimiser for checking that the dataset is of the correct input shape.
        public override int GetInputSize()
        {
            // GetOutputSize() works here because input size = output size for InputLayer objects.
            return _queue.Peek().GetOutputSize();
        }

        public override int GetOutputSize()
        {
            return _queue.PeekEnd().GetOutputSize();
        }

        public List<Layer> GetLayers()
        {
            return _layers;
        }
    }
}
