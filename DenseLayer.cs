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
        private float[] _bias;
        private int _units;
        private Activation _activationType;
        private int _inputSize;
        private WeightInitialisation _weightInitialisation;    // Weight initialisation type should be known at time of layer construction.
        private BiasInitialisation _biasInitialisation;
        // These three variables are nullable as whether they are used depends on whether there are one or more input layers.
        private Layer? _previousLayer;
        private List<Layer>? _previousLayers = new List<Layer>();
        private MergeType? _mergeType;
        // Variables used so that values can be easily retrieved for model training.
        private float[,]? _weightedOutput;
        private float[,]? _activationOutput;

        // Constructor for just one input layer.
        public DenseLayer(int units, Activation activation, WeightInitialisation weightInitialisation, BiasInitialisation biasInitialisation, Layer previousLayer)
        {
            _units = units;
            _bias = new float[units];
            _activationType = activation;
            _weightInitialisation = weightInitialisation;
            _biasInitialisation = biasInitialisation;
            _previousLayer = previousLayer;
            _inputSize = previousLayer.GetOutputSize();
            InitialiseWeights();
            InitialiseBias();
        }

        // Overloading constructor to support multiple input layers.
        public DenseLayer(int units, Activation activation, WeightInitialisation weightInitialisation, BiasInitialisation biasInitialisation, List<Layer> previousLayers, MergeType mergeType)
        {
            _units = units;
            _bias = new float[units];
            _activationType = activation;
            _weightInitialisation = weightInitialisation;
            _biasInitialisation = biasInitialisation;
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
            InitialiseWeights();    // Initialises weights regardless of merge type.
            InitialiseBias();
        }

        public override int GetOutputSize()    // Output size of one layer is the input size of the next (barring multi-branch layer topologies and batching).
        {
            return _units;
        }

        public override Activation GetActivation()
        {
            return _activationType;
        }

        public void InitialiseBias()
        {
            switch (_biasInitialisation)
            {
                case BiasInitialisation.Ones:    // Anonymous functions that initialise weights.
                    Function.Vectorise(_bias, x => 1);
                    break;
                case BiasInitialisation.Zeroes:
                    Function.Vectorise(_bias, x => 0);
                    break;
                case BiasInitialisation.Random:
                    Random rngRandom = new Random();
                    Function.Vectorise(_bias, x => (float)rngRandom.NextDouble());
                    break;
                case BiasInitialisation.Xavier:
                    Random rngXavier = new Random();
                    float lower = (float)-(1.0 / Math.Sqrt(_inputSize)), upper = (float)(1.0 / Math.Sqrt(_inputSize));
                    Function.Vectorise(_bias, x => lower + ((float)rngXavier.NextDouble() * (upper - lower)));
                    break;
            }
        }

        public void InitialiseWeights()
        {
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
        }

        // WeightedOutput methods will be used for model training, as we need this data to compute errors in each layer.
        public override float[] WeightedOutput(float[] input)
        {
            return Function.Add(_bias, Function.Multiply(_weights, input));
        }

        public float[] WeightedOutput(List<float[]> inputs)
        {
            switch (_mergeType)
            {
                case MergeType.Add:
                    float[] addInput = new float[inputs[0].Length];
                    foreach (float[] input in inputs)
                    {
                        addInput = Function.Add(addInput, input);
                    }
                    return WeightedOutput(addInput);
                case MergeType.Concatenate:
                    List<float> concatenateInput = new List<float>();
                    foreach (float[] input in inputs)
                    {
                        concatenateInput.AddRange(input);
                    }
                    return WeightedOutput(concatenateInput.ToArray());
                default:
                    throw new ArgumentException("Merge type must be either Add or Concatenate");
            }
        }

        public override float[,] WeightedOutput(float[,] input)
        {
            float[,] testWeights = this.GetWeights();
            float[,]  weightedOutput = Function.Add(Function.GetVectorAsMatrix(_bias, input.GetLength(1)), Function.Multiply(_weights, input));
            SetWeightedOutput(weightedOutput);
            return weightedOutput;
        }

        public float[,] WeightedOutput(List<float[,]> inputs)
        {
            switch (_mergeType)
            {
                case MergeType.Add:
                    float[,] addInput = new float[inputs[0].GetLength(0), inputs[0].GetLength(1)];
                    foreach (float[,] input in inputs)
                    {
                        addInput = Function.Add(addInput, input);
                    }
                    return WeightedOutput(addInput);
                case MergeType.Concatenate:
                    return WeightedOutput(Function.RowConcatenate(inputs)); 
                default:
                    // Default shouldn't be checked because all calls to this method should be for merging layers.
                    throw new ArgumentException("Merge type must be either Add or Concatenate");
            }
        }

        public override float[] ForwardPass(float[] input)
        {
            if (input.Length != _inputSize)
            {
                throw new ArgumentException("Input length doesn't equal input size of layer");
            }
            float[] output = WeightedOutput(input);
            switch (_activationType)
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
                    Function.Vectorise(output, Function.None);
                    return output;
                default:
                    throw new ArgumentException("Activation must be one of the activation functions specified in the enumeration");
            }
        }

        // Overloading ForwardPass to support a list of inputs, used when the Dense layer is merging outputs from multiple previous layers.
        public float[] ForwardPass(List<float[]> inputs)
        {
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
            float[,] output = new float[inputs.GetLength(0),inputs.GetLength(1)];
            output = WeightedOutput(inputs);
            switch (_activationType)
            {
                case Activation.ReLU:
                    Function.Vectorise(output, Function.ReLU);
                    break;
                case Activation.Sigmoid:
                    Function.Vectorise(output, Function.Sigmoid);
                    break;
                case Activation.SiLU:
                    Function.Vectorise(output, Function.SiLU);
                    break;
                case Activation.Tanh:
                    Function.Vectorise(output, Function.Tanh);
                    break;
                case Activation.None:
                    Function.Vectorise(output, Function.None);
                    break;
                default:
                    throw new ArgumentException("Activation must be one of the enumerated types");
            }
            SetActivationOutput(output);
            return output;
        }

        // Used for merge phase with mini-batch stochastic gradient descent.
        public float[,] ForwardPass(List<float[,]> inputs)
        {
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
                    output = ForwardPass(Function.RowConcatenate(inputs));
                    break;
                default:
                    throw new ArgumentException("Merge type must be either Add or Concatenate");
                    // Default shouldn't be checked because all calls to this method should be for merging layers.
            }
            return output;
        }

        // These two functions GetWeightedOutput() and GetActivationOutput() get the *actual* computed values of the weighted output and activation output.
        // This is for use in model training.
        public override float[,] GetWeightedOutput()
        {
            return _weightedOutput;
        }

        public override float[,] GetActivationOutput()
        {
            return _activationOutput;
        }

        public void SetWeightedOutput(float[,] weightedOutput)
        {
            _weightedOutput = weightedOutput;
        }

        public void SetActivationOutput(float[,] activationOutput)
        {
            _activationOutput = activationOutput;
        }

        public void ModifyWeights(float[,] modification)
        {
            Function.Add(_weights, modification);
        }

        public float[,] GetWeights()
        {
            return _weights;
        }

        public float[] GetWeightsDimension()
        {
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
