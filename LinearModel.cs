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
        List<Layer> layers = new List<Layer>();
        // Nullable until constructor called.
        Queue<Layer>? queue;
        // Boolean that indicates whether the model is ready to use and train - Compile() is a prerequisite.
        bool _readyToTrain = false;
        int _batchSize;
        CostFunction _costFunction;
        float _learningRate;
        int _epochs;

        public override void AddLayer(Layer layer)
        {
            layers.Add(layer);
        }

        public override float[] ForwardPropagate(float[] input)
        {
            if (_readyToTrain == false)
            {
                throw new Exception("Model must be compiled before training");
            }
            else
            {
                for (int i = 0; i < queue.GetQueueSize(); i++)
                {
                    input = queue.Dequeue().ForwardPass(input);
                }
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
                for (int i = 0; i < queue.GetQueueSize(); i++)
                {
                    input = queue.Dequeue().ForwardPass(input);
                }
                return input;
            }
        }

        public override void Compile(CostFunction costFunction)
        {
            _costFunction = costFunction;

            if (layers[0] is not InputLayer)
            {
                throw new Exception("First layer must be Input layer");
            }
            if (layers.FindAll(s => s is InputLayer).Count > 1)
            {
                throw new Exception("Only one input layer is allowed");
            }

            queue = new Queue<Layer>(layers.Count());
            foreach (Layer l in layers)
            {
                queue.Enqueue(l);
            }
            _readyToTrain = true;
        }

        public override void Train(string filename, int epochs = 10, float learningRate = 0.1f, int batchSize = 1)
        {
            if (_readyToTrain == false)
            {
                throw new Exception("Model must be compiled before training");
            }
            _epochs = epochs;
            _learningRate = learningRate;
            _batchSize = batchSize;

            Optimiser.Fit(this, filename); // Implement and overload optimiser for Linear and Functional/Complex/multi-branch models.
            throw new NotImplementedException();
        }

        // public override Train for reading data off a text file too.

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
            return queue.Peek().GetOutputSize();
        }

        public override int GetOutputSize()
        {
            return queue.PeekEnd().GetOutputSize();
        }
    }
}
