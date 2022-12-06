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

        public override void AddLayer(Layer layer)
        {
            _layers.Add(layer);
        }

        public List<Layer> GetLayers()
        {
            return _layers;
        }

        // float[] is the output of the model.
        public override float[] ForwardPropagate(float[] input)
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
                throw new Exception("First layer must be Input layer");
            }
            if (_layers.FindAll(s => s is InputLayer).Count > 1)
            {
                throw new Exception("Only one input layer is allowed");
            }

            _queue = new Queue<Layer>(_layers.Count());
            foreach (Layer l in _layers)
            {
                _queue.Enqueue(l);
            }
            _readyToTrain = true;
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
            return _queue.Peek().GetOutputSize();
        }

        public override int GetOutputSize()
        {
            return _queue.PeekEnd().GetOutputSize();
        }
    }
}
