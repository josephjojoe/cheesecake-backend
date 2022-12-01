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
        Stack<DenseLayer>? _stack;    // We make sure that the stack only contains DenseLayer objects.
        // Boolean that indicates whether the model is ready to use and train - Compile() is a prerequisite.
        bool _readyToTrain = false;
        int _batchSize;
        CostFunction _costFunction;
        float _learningRate;
        int _epochs;

        public override void AddLayer(Layer layer)
        {
            _layers.Add(layer);
        }

        // float[] is the output of the model, List<float[]> contains weighted inputs for each layer.
        public override Tuple<float[], List<float[]>> ForwardPropagate(float[] input)
        {
            if (_readyToTrain == false)
            {
                throw new Exception("Model must be compiled before training");
            }
            else
            {
                List<float[]> weightedOutputs = new List<float[]>();
                for (int i = 0; i < _queue.GetQueueSize(); i++)
                {
                    Layer layer = _queue.Dequeue();
                    if (layer is DenseLayer)    // Doesn't add InputLayer weighted output to List.
                    {
                        weightedOutputs.Add(layer.WeightedOutput(input));
                        _stack.Push((DenseLayer)layer);    // Pushes layer (Dense) onto the stack for future retrieval and training.
                    }
                    input = layer.ForwardPass(input);
                }
                Requeue();
                return new Tuple<float[], List<float[]>>(input, weightedOutputs);
            }
        }

        // Polymorphism for batched inputs.
        public override Tuple<float[,], List<float[,]>> ForwardPropagate(float[,] input)
        {
            if (_readyToTrain == false)
            {
                throw new Exception("Model must be compiled before training");
            }
            else
            {
                List<float[,]> weightedOutputs = new List<float[,]>();
                for (int i = 0; i < _queue.GetQueueSize(); i++)
                {
                    Layer layer = _queue.Dequeue();
                    if (layer is DenseLayer)    // Doesn't add InputLayer weighted output to List.
                    {
                        weightedOutputs.Add(layer.WeightedOutput(input));
                        _stack.Push((DenseLayer)layer);
                    }
                    input = layer.ForwardPass(input);
                }
                Requeue();
                return new Tuple<float[,], List<float[,]>>(input, weightedOutputs);
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
                throw new Exception("First layer must be Input layer");
            }
            if (_layers.FindAll(s => s is InputLayer).Count > 1)
            {
                throw new Exception("Only one input layer is allowed");
            }

            // Stack doesn't include InputLayer
            _stack = new Stack<DenseLayer>(_layers.Count() - 1);
            _queue = new Queue<Layer>(_layers.Count());
            foreach (Layer l in _layers)
            {
                _queue.Enqueue(l);
            }
            _readyToTrain = true;
        }

        public Stack<DenseLayer> GetStack()
        {
            return _stack;
        }

        public override void Train(string filename, int epochs = 10, float learningRate = 0.1f, int batchSize = 1)
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

            Optimiser.Fit(this, filename); // Implement and overload optimiser for Linear and Functional/Complex/multi-branch models.
            throw new NotImplementedException();
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
            return _queue.Peek().GetOutputSize();
        }

        public override int GetOutputSize()
        {
            return _queue.PeekEnd().GetOutputSize();
        }
    }
}
