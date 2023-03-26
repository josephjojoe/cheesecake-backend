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

        public override float[,] ForwardPropagate(float[,] inputs)
        {
            if (_readyToTrain == false)
            {
                throw new Exception("Model must be compiled before training");
            }
            if (graph.GetEndNodes(graph.GetTopology()).Count > 1)
            {
                throw new Exception("Multiple output layers disallowed.");
            }
            List<Layer> order = graph.GetTopologicalSort();
            // Sets up activation output on first layer (the input layer).
            inputs = order[0].ForwardPass(inputs);
            // Starts from 1 as we are excluding an explicit ForwardPass call on the first layer (the input layer).
            for (int i = 1; i < order.Count; i++)
            {
                // Collects inputs from incoming layers.
                List<Layer> incomingLayers = graph.GetIncomingNodes(graph.GetTopology(), order[i]);
                List<float[,]> incomingInputs = new List<float[,]>();
                foreach (Layer l in incomingLayers)
                {
                    incomingInputs.Add(l.GetActivationOutput());
                }
                DenseLayer denseLayer = (DenseLayer)order[i];
                // Updates activation output for layers as we traverse the graph.
                float[,] output = denseLayer.ForwardPass(incomingInputs);
            }
            // Model output is the activation output of the final layer.
            DenseLayer outputLayer = (DenseLayer)order.Last();
            return outputLayer.GetActivationOutput();
        }

        public override void Compile(CostFunction costFunction)
        {
            _costFunction = costFunction;
            _readyToTrain = true;
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