using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public class DenseLayer : Layer
    {
        private float[,]? _weights;    // Weights are null until initialised with model compilation, as input data shape must be known.
        private bool _weightsInitialised = false;    // Flag to check if weights are initialised yet (must be done before forward pass), prevents double initialisation.
        private float[] _bias;
        private int _units;
        private Activation _activation;
        private int _inputSize;
        private WeightInitialisation _weightInitialisation;    // Weight initialisation type should be known at time of layer construction.
        // These three variables are nullable as whether they are used depends on whether there are one or more input layers.
        private Layer? _previousLayer;
        private List<Layer>? _previousLayers = new List<Layer>();
        private MergeType? _mergeType;

        // Constructor for just one input layer.
        public DenseLayer(int units, Activation activation, WeightInitialisation weightInitialisation, Layer previousLayer)
        {
            _units = units;
            _bias = new float[units];
            _activation = activation;
            _weightInitialisation = weightInitialisation;
            _previousLayer = previousLayer;
            _inputSize = previousLayer.GetOutputSize();
            InitialiseWeights(_inputSize);
        }

        // Overloading constructor to support multiple input layers.
        public DenseLayer(int units, Activation activation, WeightInitialisation weightInitialisation, List<Layer> previousLayers, MergeType mergeType)
        {
            _units = units;
            _bias = new float[units];
            _activation = activation;
            _weightInitialisation = weightInitialisation;
            _previousLayers = previousLayers;
            if (mergeType == MergeType.Add)
            {
                List<int> outputShapes = new List<int>();
                foreach (Layer l in previousLayers)
                {
                    outputShapes.Add(l.GetOutputSize());
                }
                if (outputShapes.Distinct().Count() == 1)
                {
                    _mergeType = mergeType;
                    // For Add, input size to this layer is the same as the output size of any one of the input layers.
                    _inputSize = outputShapes[0];
                }
                else
                {
                    // Throw some error - for Add, output shapes of all branches must match.
                }
            }
            if (mergeType == MergeType.Concatenate)
            {
                _mergeType = mergeType;
                foreach (Layer l in previousLayers)
                {
                    _inputSize += l.GetOutputSize();    // Output size with concatenate is the sum of all input branch output dimensions.
                }
            }
            InitialiseWeights(_inputSize);    // Initialises weights regardless of merge type.
        }

        public override int GetOutputSize()    // Output size of one layer is the input size of the next (barring multi-branch layer topologies and batching).
        {
            return _units;
        }

        public Activation GetActivation()
        {
            return _activation;
        }

        public void InitialiseWeights(int inputSize)
        {
            if (_weightsInitialised == true)
            {
                // Throw some error - weights should not be initialised twice for a layer.
            }
            _inputSize = inputSize;
            _weights = new float[_units, _inputSize];    // Is this the right way around for the weights matrix? Make sure to test!
            switch (_weightInitialisation)
            {
                case WeightInitialisation.Ones:    // Anonymous functions that initialise weights.
                    Function.Vectorise(_weights, x => 1);
                    break;
                case WeightInitialisation.Zeroes:
                    Function.Vectorise(_weights, x => 0);
                    break;
                case WeightInitialisation.Random:
                    Random rngRandom = new Random();
                    Function.Vectorise(_weights, x => (float)rngRandom.NextDouble());
                    break;
                case WeightInitialisation.Xavier:
                    Random rngXavier = new Random();
                    float lower = (float)-(1.0 / Math.Sqrt(_inputSize)), upper = (float)(1.0 / Math.Sqrt(_inputSize));
                    Function.Vectorise(_weights, x => lower + ((float)rngXavier.NextDouble() * (upper - lower)));
                    break;
            }
            _weightsInitialised = true;    // Set weights initialised to true after this function is called.
        }

        public override float[] ForwardPass(float[] input)
        {
            if (input.Length != _inputSize)
            {
                // Throw some error.
            }
            if (_weightsInitialised == false)
            {
                // Throw another error.
            }

            float[] output = new float[_inputSize];
            output = Function.Multiply(_weights, input);

            switch (_activation)
            {
                case Activation.ReLU:
                    Function.Vectorise(output, Function.ReLU);
                    return output;
                case Activation.Sigmoid:
                    Function.Vectorise(output, Function.Sigmoid);
                    return output;
                case Activation.SiLU:
                    Function.Vectorise(output, Function.SiLU);
                    return output;
                case Activation.Tanh:
                    Function.Vectorise(output, Function.Tanh);
                    return output;
                case Activation.None:
                    break;    // Returns output outside of switch-case so that all code paths return a value.
            }
            return output;
        }

        // Overloading ForwardPass to support a list of inputs, used when the Dense layer is merging outputs from multiple previous layers.
        public float[] ForwardPass(List<float[]> inputs)
        {
            if (_weightsInitialised == false)
            {
                // Throw some error - weights must be initialised.
            }
            float[] output;
            switch (_mergeType)
            {
                case MergeType.Add:
                    // Constructor has checked for the "Add" merge type that all of these inputs are of the same length.
                    // Hence here we do not need error checking for length of inputs, we just need to implement proper processsing.
                    float[] addInput = new float[inputs[0].Length];
                    foreach (float[] input in inputs)
                    {
                        addInput = Function.Add(addInput, input);
                    }
                    output = ForwardPass(addInput);
                    break;
                case MergeType.Concatenate:
                    List<float> concatenateInput = new List<float>();
                    foreach (float[] input in inputs)
                    {
                        concatenateInput.AddRange(input);
                    }
                    output = ForwardPass(concatenateInput.ToArray());
                    break;
                default:
                    // Throw some error - method shouldn't be used if not merging multiple branches of a layer topology.
                    throw new NotImplementedException();
            }
            return output;
        }

        // Used when performing forward pass on a batched set of vectors (grouped in the form of a matrix).
        public override float[,] ForwardPass(float[,] inputs)
        {
            if (_weightsInitialised == false)
            {
                // Throw some error - weights must be initialised.
            }
            float[,] output = new float[inputs.GetLength(0),inputs.GetLength(1)];
            output = Function.Multiply(_weights, inputs);
            switch (_activation)
            {
                case Activation.ReLU:
                    Function.Vectorise(output, Function.ReLU);
                    return output;
                case Activation.Sigmoid:
                    Function.Vectorise(output, Function.Sigmoid);
                    return output;
                case Activation.SiLU:
                    Function.Vectorise(output, Function.SiLU);
                    return output;
                case Activation.Tanh:
                    Function.Vectorise(output, Function.Tanh);
                    return output;
                case Activation.None:
                    break;    // Returns output outside of switch-case so that all code paths return a value.
            }
            return output;
        }

        // Used for merge phase with mini-batch stochastic gradient descent.
        public float[,] ForwardPass(List<float[,]> inputs)
        {
            if (_weightsInitialised == false)
            {
                // Throw some error - weights must be initialised.
            }
            float[,] output;
            switch (_mergeType)
            {
                // Note to self - check that DenseLayer constructor allows for this?
                case MergeType.Add:
                    float[,] addInput = new float[inputs[0].GetLength(0), inputs[0].GetLength(1)];
                    foreach (float[,] input in inputs)
                    {
                        addInput = Function.Add(addInput, input);
                    }
                    output = ForwardPass(addInput);
                    break;
                case MergeType.Concatenate:
                    output = ForwardPass(Function.Concatenate(inputs));
                    break;
                default:
                    throw new NotImplementedException();
                    // Default shouldn't be checked because all calls to this method should be for merging layers.
            }
            return output;
        }

        public void ModifyWeights(float[,] modification)
        {
            if (_weightsInitialised == false)
            {
                // Throw some error - weights must be initialised.
            }
            Function.Add(_weights, modification);
        }

        public float[,] GetWeights()
        {
            if (_weightsInitialised == false)
            {
                // Throw some error - weights must be initialised.
            }
            return _weights;
        }

        public float[] GetWeightsDimension()
        {
            if (_weightsInitialised == false)
            {
                // Throw some error - weights must be initialised.
            }
            return new float[2] { _weights.GetLength(0), _weights.GetLength(1) };
        }

        public void ModifyBias(float[] modification)
        {
            Function.Add(_bias, modification);
        }

        public float[] GetBias()
        {
            return _bias;
        }

        public int GetUnits()
        {
            return _units;
        }
    }
}
