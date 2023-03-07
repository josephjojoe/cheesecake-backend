using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public class ComplexModel : Model
    {
        DirectedAcyclicGraph graph = new DirectedAcyclicGraph();
        bool _inputLayerAdded = false;
        bool _readyToTrain = false;
        int _batchSize;
        CostFunction _costFunction;
        float _learningRate;
        int _epochs;

        public void AddInputLayer(InputLayer layer)
        {
            if (_inputLayerAdded == false)
            {
                graph.AddInputNode(layer);
                _inputLayerAdded = true;
            }
            else
            {
                throw new ArgumentException("Input layer has already been added.");
            }
        }

        public void AddLayer(Layer previousLayer, Layer layer)
        {
            graph.AddConnection(previousLayer, layer);
        }

        public override int GetInputSize()
        {
            // Redundancy for multiple input layers in tentative future development.
            List<Layer> inputLayers = graph.GetStartNodes(graph.GetTopology());
            return inputLayers[0].GetOutputSize();
        }

        public override int GetOutputSize()
        {
            return graph.GetTopologicalSort().Last().GetOutputSize();
        }

        // This isn't how forward propagate works in complex models!!!
        public override float[] ForwardPropagate(float[] input)
        {
            List<Layer> order = graph.GetTopologicalSort();
            foreach (Layer layer in order)
            {
                input = layer.ForwardPass(input);
            }
            return input;
        }

        // This also isn't how forward propagate works in complex models!!!
        public override float[,] ForwardPropagate(float[,] input)
        {
            List<Layer> order = graph.GetTopologicalSort();
            foreach (Layer layer in order)
            {
                input = layer.ForwardPass(input);
            }
            return input;
        }

        public override void Compile(CostFunction costFunction)
        {
            _costFunction = costFunction;
            _readyToTrain = true;
        }

        public override void Train(string filename, int epochs, float learningRate, int batchSize)
        {
            _epochs = epochs;
            _learningRate = learningRate;
            _batchSize = batchSize;
            Optimiser.Fit(this, filename);
        }

        public Dictionary<Layer, List<Layer>> GetTopology()
        {
            return graph.GetTopology();
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
    }
}
