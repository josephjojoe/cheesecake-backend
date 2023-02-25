using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    // Directed acyclic graph class for holding multi-branch layer topologies - used in the ComplexModel class.
    // We utilise the adjacency list approach and store connections as key-value sender-receiver pairs, e.g. {input_layer: dense_layer_1}.
    public class DirectedAcyclicGraph
    {
        // Nullable because layers will only be added once AddLayer() in ComplexModel is called.
        // Topology of layers and a list of layers that the layer connects to.
        Dictionary<Layer, List<Layer>> _topology = new Dictionary<Layer, List<Layer>>();
        // List corresponding to the topological sort order in which layers should be accessed during the forward pass.
        List<Layer> _topologicalSortOrder = new List<Layer>();
        // Safeguard which ensures only one input node is present in the graph.
        bool _inputNodeAdded = false;
        
        public void AddInputNode(Layer inputLayer)
        {
            if (_inputNodeAdded == false)
            {
                _topology.Add(inputLayer, new List<Layer>());
                _inputNodeAdded = true;
                _topologicalSortOrder.Add(inputLayer);
            }
            else
            {
                throw new ArgumentException("Single input node already added to graph.");
            }
        }

        public void AddConnection(Layer layer1, Layer layer2)
        {
            if (!_topology.ContainsKey(layer1))
            {
                _topology.Add(layer1, new List<Layer>() { layer2 });
            }
            else
            {
                List<Layer> linkedLayers = _topology[layer1];
                if (!linkedLayers.Contains(layer2))
                {
                    linkedLayers.Add(layer2);
                    _topology[layer1] = linkedLayers;
                }
                else
                {
                    throw new ArgumentException("Connection already exists between the specified nodes. Multiple edges between nodes is not supported.");
                }
            }
            // Dynamically updates traversal order as new nodes are added.
            // TopologicalSort();
        }

        public void TopologicalSort()
        {
            Dictionary<Layer, List<Layer>> topology = _topology;

            // Implementation of Kahn's topological sort algorithm.
            // Provides a linearisation of the graph topology (order in which each layer should be visited during the forward pass)
            // and conveniently checks that the graph is acyclic during execution.
            List<Layer> L = new List<Layer>();
            List<Layer> S = GetStartNodes(topology);
            while (S.Count != 0)
            {
                Layer node = S[0];
                S.RemoveAt(0);
                L.Add(node);

                List<Layer> outgoingNodes = GetOutgoingNodes(topology, node);

                // Removes all directed edges from L to other layers.
                topology[node] = new List<Layer>();

                for (int i = 0; i < outgoingNodes.Count; i++)
                {
                    if (GetIncomingNodes(topology, outgoingNodes[i]).Count == 0)
                    {
                        S.Add(outgoingNodes[i]);
                    }
                }
            }
            for (int i = 0; i < topology.Keys.ToArray().Length; i++)
            {
                if (topology[topology.Keys.ToArray()[i]].Count != 0)
                {
                    throw new ArgumentException("Graph contains a cycle.");
                }
            }
            _topologicalSortOrder = L;
        }

        public List<Layer> GetTopologicalSort()
        {
            return _topologicalSortOrder;
        }

        public Dictionary<Layer, List<Layer>> GetTopology()
        {
            return _topology;
        }

        // Returns a list of the nodes that have directed connections terminating at the supplied node.
        // In 'layer' terms, which layers feed into this one?
        public List<Layer> GetIncomingNodes(Dictionary<Layer, List<Layer>> topology, Layer layer)
        {
            // Returns *generic* set of values from a dictionary - will be reused for Kahn's topological sort algorithm.
            List<Layer> incomingNodes = new List<Layer>();
            for (int i = 0; i < topology.Keys.ToArray().Length; i++)
            {
                if (topology[topology.Keys.ToArray()[i]].Contains(layer))
                {
                    incomingNodes.Add(topology.Keys.ToArray()[i]);
                }
            }
            return incomingNodes;
        }

        // Returns a list of the nodes that this node supplies directed connections to.
        // In 'layer' terms, which layers does this layer feed into?
        public List<Layer> GetOutgoingNodes(Dictionary<Layer, List<Layer>> topology, Layer layer)
        {
            try
            {
                return topology[layer];
            }
            catch
            {
                return new List<Layer>();
            }
        }

        // Returns start nodes - nodes that do not have any incoming nodes. For our DAG constraints, this list should contain just one layer.
        // For the purpose of code reuse, however, my algorithm implementation should be able to support a topological sort of DAGs containing multiple.
        // This also corresponds to all InputLayer instances in the DAG - both conditions are checked as a failsafe.
        public List<Layer> GetStartNodes(Dictionary<Layer, List<Layer>> topology)
        {
            List<Layer> startNodes = new List<Layer>();
            for (int i = 0; i < topology.Keys.ToArray().Length; i++)
            {
                if (GetIncomingNodes(topology, topology.Keys.ToArray()[i]).Count == 0)
                {
                    if (topology.Keys.ToArray()[i] is InputLayer)
                    {
                        startNodes.Add(topology.Keys.ToArray()[i]);
                    }
                    else
                    {
                        // If a start node is a dense layer, this means that the graph is disconnected.
                        throw new ArgumentException("Graph has disconnected components.");
                    }
                }
            }
            return startNodes;
        }
    }
}
