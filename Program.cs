﻿using System.ComponentModel;

namespace Backend
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Landing();
        }

        // Generic option chooser utility function that returns the index of the chosen option.
        static int ChooseOption(List<string> options, string promptText = "Choose option:")
        {
            for (int i = 0; i < options.Count; i++)
            {
                Console.WriteLine($"({i}) {options[i]}");
            }
            int chosenOption = -1;
            do
            {
                Console.WriteLine(promptText);
                try
                {
                    chosenOption = int.Parse(Console.ReadLine());
                }
                catch
                {
                    Console.WriteLine("Invalid value.");
                }
            }
            // Keeps loop ongoing while a valid choice hasn't been made.
            while (!Enumerable.Range(0, options.Count).ToList().Contains(chosenOption));
            return chosenOption;
        }

        static void Landing()
        {
            Console.WriteLine("Welcome to Cheesecake! A programming environment for AI model building in C#.");
            List<string> modelType = new List<string>() { "Linear model (training)", "Non-linear model (demonstration)"};
            int modelTypeChoice = ChooseOption(modelType, "Choose model type:");
            switch (modelTypeChoice)
            {
                case 0:
                    LinearModelBuilder();
                    break;
                case 1:
                    NonLinearModelBuilder();
                    break;
                default:
                    // Default should never be executed because of the data validation in ChooseOption(), but a default for switch-case is good practise.
                    LinearModelBuilder();
                    break;
            }
        }

        // Allows the user to build a linear model.
        // Relatively self-documenting/explanatory code.
        static void LinearModelBuilder()
        {
            LinearModel model = new LinearModel();
            List<Layer> layers = new List<Layer>();

            int inputSize = 0;
            do
            {
                Console.WriteLine("Enter model input size:");
                try
                {
                    inputSize = int.Parse(Console.ReadLine());
                }
                catch
                {
                    Console.WriteLine("Invalid value.");
                }
            }
            while (inputSize < 1);

            InputLayer layer = new InputLayer(inputSize);
            layers.Add(layer);
            model.AddLayer(layer);

            bool addMoreLayers = true;
            while (addMoreLayers == true)
            {
                int units = -1;
                do
                {
                    Console.WriteLine("Enter dense layer units:");
                    try
                    {
                        units = int.Parse(Console.ReadLine());
                    }
                    catch
                    {
                        Console.WriteLine("Invalid value.");
                    }
                }
                while (units < 1);

                Activation? activation = null;
                do
                {
                    int activationChoice = ChooseOption(new List<String>() { "ReLU", "Sigmoid", "Tanh", "SiLU", "None" },
                        "Choose activation function:");
                    switch (activationChoice)
                    {
                        case 0:
                            activation = Activation.ReLU;
                            break;
                        case 1:
                            activation = Activation.Sigmoid;
                            break;
                        case 2:
                            activation = Activation.Tanh;
                            break;
                        case 3:
                            activation = Activation.SiLU;
                            break;
                        case 4:
                            activation = Activation.None;
                            break;
                        default:
                            // Should never occur, but having a default case is good C# practice for switch-case.
                            activation = Activation.None;
                            break;
                    }
                }
                while (activation == null);

                WeightInitialisation? weightInitialisation = null;
                do
                {
                    int weightInitialisationChoice = ChooseOption(new List<String>() { "Zeroes", "Ones", "Random", "Xavier"},
                        "Choose weight initialisation function:");
                    switch (weightInitialisationChoice)
                    {
                        case 0:
                            weightInitialisation = WeightInitialisation.Zeroes;
                            break;
                        case 1:
                            weightInitialisation = WeightInitialisation.Ones;
                            break;
                        case 2:
                            weightInitialisation = WeightInitialisation.Random;
                            break;
                        case 3:
                            weightInitialisation = WeightInitialisation.Xavier;
                            break;
                        default:
                            // Should never occur, but having a default case is good C# practice for switch-case.
                            weightInitialisation = WeightInitialisation.Xavier;
                            break;
                    }
                }
                while (weightInitialisation == null);

                BiasInitialisation? biasInitialisation = null;
                do
                {
                    int biasInitialisationChoice = ChooseOption(new List<String>() { "Zeroes", "Ones", "Random", "Xavier" },
                        "Choose bias initialisation function:");
                    switch (biasInitialisationChoice)
                    {
                        case 0:
                            biasInitialisation = BiasInitialisation.Zeroes;
                            break;
                        case 1:
                            biasInitialisation = BiasInitialisation.Ones;
                            break;
                        case 2:
                            biasInitialisation = BiasInitialisation.Random;
                            break;
                        case 3:
                            biasInitialisation = BiasInitialisation.Xavier;
                            break;
                        default:
                            // Should never occur, but having a default case is good C# practice for switch-case.
                            biasInitialisation = BiasInitialisation.Xavier;
                            break;
                    }
                }
                while (biasInitialisation == null);

                // Explicit casts needed as activation, weight initialisation, and bias initialisation are nullable.
                DenseLayer newLayer = new DenseLayer(units, (Activation)activation,
                    (WeightInitialisation)weightInitialisation, (BiasInitialisation)biasInitialisation, layers.Last());
                layers.Add(newLayer);
                model.AddLayer(newLayer);

                Console.WriteLine("Add more layers? (Y/N)");
                string response = Console.ReadLine();
                if (response.ToLower() == "y")
                {
                    continue;
                }
                else
                {
                    addMoreLayers = false;
                }
            }

            Console.WriteLine("Would you like to load layer parameters from a text file? (Y/N)");
            string loadLayerParametersResponse = Console.ReadLine();
            while (loadLayerParametersResponse.ToLower() == "y")
            {
                Console.WriteLine("What is the index of the layer you would like to load layer parameters into?");
                int index = -1;
                do
                {
                    try
                    {
                        index = int.Parse(Console.ReadLine());
                        if (!Enumerable.Range(0, layers.Count).ToList().Contains(index))
                        {
                            // For input integers which aren't in the correct range.
                            Console.WriteLine("Invalid number.");
                        }
                    }
                    catch
                    {
                        // For inputs which are of invalid type.
                        Console.WriteLine("Invalid number.");
                    }
                }
                while (!Enumerable.Range(0, layers.Count).ToList().Contains(index));

                Console.WriteLine("What is the file name of the parameters file?");
                string parametersFileName = Console.ReadLine();
                try
                {
                    ((DenseLayer)layers[index]).LoadWeightsAndBias(parametersFileName);
                }
                catch
                {
                    Console.WriteLine("Incompatible file.");
                }

                Console.WriteLine("Would you like to load layer parameters into another layer? (Y/N)");
                loadLayerParametersResponse = Console.ReadLine();
            }

            CostFunction? costFunction = null;
            do
            {
                int costFunctionChoice = ChooseOption(new List<String>() { "MSE", "MAE" },
                        "Choose cost function:");
                switch (costFunctionChoice)
                {
                    case 0:
                        costFunction = CostFunction.MSE;
                        break;
                    case 1:
                        costFunction = CostFunction.MAE;
                        break;
                    default:
                        // Should never occur, but having a default case is good C# practice for switch-case.
                        costFunction = CostFunction.MSE;
                        break;
                }
            }
            while (costFunction == null);
            model.Compile((CostFunction)costFunction);

            int epochs = -1;
            do
            {
                Console.WriteLine("Enter number of epochs to train for:");
                try
                {
                    epochs = int.Parse(Console.ReadLine());
                }
                catch
                {
                    Console.WriteLine("Invalid value.");
                }
            }
            while (epochs < 1);

            float learningRate = -1;
            do
            {
                Console.WriteLine("Enter learning rate:");
                try
                {
                    learningRate = float.Parse(Console.ReadLine());
                }
                catch
                {
                    Console.WriteLine("Invalid value.");
                }
            }
            while (learningRate <= 0);

            int batchSize = -1;
            do
            {
                Console.WriteLine("Enter batch size:");
                try
                {
                    batchSize = int.Parse(Console.ReadLine());
                }
                catch
                {
                    Console.WriteLine("Invalid value.");
                }
            }
            while (batchSize < 1);

            Console.WriteLine("Enter dataset filename:");
            string datasetFileName = Console.ReadLine();
            if (!datasetFileName.EndsWith(".txt"))
            {
                datasetFileName += ".txt";
            }

            Tuple<List<float[]>, List<float[]>>? dataset = null;
            while (dataset == null)
            {
                try
                {
                    dataset = Data.ExtractDataset(datasetFileName, layers[0].GetOutputSize(), layers.Last().GetOutputSize());
                }
                catch
                {
                    Console.WriteLine("Invalid dataset. Enter dataset filename:");
                    datasetFileName = Console.ReadLine();
                }
            }

            try
            {
                model.Train(datasetFileName, epochs, learningRate, batchSize);
            }
            catch
            {
                Console.WriteLine("Invalid dataset and/or parameters.");
            }

            Console.WriteLine("Would you like to save layer parameters to a text file? (Y/N)");
            string saveLayerParametersResponse = Console.ReadLine();
            while (saveLayerParametersResponse.ToLower() == "y")
            {
                Console.WriteLine("What is the index of the layer you would like to save layer parameters from?");
                int index = -1;
                do
                {
                    try
                    {
                        index = int.Parse(Console.ReadLine());
                        if (!Enumerable.Range(0, layers.Count).ToList().Contains(index))
                        {
                            // For input integers which aren't in the correct range.
                            Console.WriteLine("Invalid number.");
                        }
                    }
                    catch
                    {
                        // For inputs which are of invalid type.
                        Console.WriteLine("Invalid number.");
                    }
                }
                while (!Enumerable.Range(0, layers.Count).ToList().Contains(index));

                Console.WriteLine("Enter filename to save parameters into:");
                string file = Console.ReadLine();
                try
                {
                    ((DenseLayer)model.GetLayers()[index]).SaveWeightsAndBias(file);
                    Console.WriteLine("Parameters successfully saved.");
                }
                catch
                {
                    Console.WriteLine("Invalid filename.");
                }
                
                Console.WriteLine("Would you like to load layer parameters into another layer? (Y/N)");
                saveLayerParametersResponse = Console.ReadLine();
            }
        }

        // Allows the user to build a non-linear model and run it in inference on vectors.
        static void NonLinearModelBuilder()
        {
            ComplexModel model = new ComplexModel();
            List<Layer> layers = new List<Layer>();

            int inputSize = 0;
            do
            {
                Console.WriteLine("Enter model input size:");
                try
                {
                    inputSize = int.Parse(Console.ReadLine());
                }
                catch
                {
                    Console.WriteLine("Invalid value.");
                }
            }
            while (inputSize < 1);

            InputLayer inputLayer = new InputLayer(inputSize);
            layers.Add(inputLayer);
            model.AddInputLayer(inputLayer);

            bool addMoreLayers = true;
            while (addMoreLayers == true)
            {
                int units = -1;
                do
                {
                    Console.WriteLine("Enter dense layer units:");
                    try
                    {
                        units = int.Parse(Console.ReadLine());
                    }
                    catch
                    {
                        Console.WriteLine("Invalid value.");
                    }
                }
                while (units < 1);

                Activation? activation = null;
                do
                {
                    int activationChoice = ChooseOption(new List<String>() { "ReLU", "Sigmoid", "Tanh", "SiLU", "None" },
                        "Choose activation function:");
                    switch (activationChoice)
                    {
                        case 0:
                            activation = Activation.ReLU;
                            break;
                        case 1:
                            activation = Activation.Sigmoid;
                            break;
                        case 2:
                            activation = Activation.Tanh;
                            break;
                        case 3:
                            activation = Activation.SiLU;
                            break;
                        case 4:
                            activation = Activation.None;
                            break;
                        default:
                            // Should never occur, but having a default case is good C# practice for switch-case.
                            activation = Activation.None;
                            break;
                    }
                }
                while (activation == null);

                WeightInitialisation? weightInitialisation = null;
                do
                {
                    int weightInitialisationChoice = ChooseOption(new List<String>() { "Zeroes", "Ones", "Random", "Xavier" },
                        "Choose weight initialisation function:");
                    switch (weightInitialisationChoice)
                    {
                        case 0:
                            weightInitialisation = WeightInitialisation.Zeroes;
                            break;
                        case 1:
                            weightInitialisation = WeightInitialisation.Ones;
                            break;
                        case 2:
                            weightInitialisation = WeightInitialisation.Random;
                            break;
                        case 3:
                            weightInitialisation = WeightInitialisation.Xavier;
                            break;
                        default:
                            // Should never occur, but having a default case is good C# practice for switch-case.
                            weightInitialisation = WeightInitialisation.Xavier;
                            break;
                    }
                }
                while (weightInitialisation == null);

                BiasInitialisation? biasInitialisation = null;
                do
                {
                    int biasInitialisationChoice = ChooseOption(new List<String>() { "Zeroes", "Ones", "Random", "Xavier" },
                        "Choose bias initialisation function:");
                    switch (biasInitialisationChoice)
                    {
                        case 0:
                            biasInitialisation = BiasInitialisation.Zeroes;
                            break;
                        case 1:
                            biasInitialisation = BiasInitialisation.Ones;
                            break;
                        case 2:
                            biasInitialisation = BiasInitialisation.Random;
                            break;
                        case 3:
                            biasInitialisation = BiasInitialisation.Xavier;
                            break;
                        default:
                            // Should never occur, but having a default case is good C# practice for switch-case.
                            biasInitialisation = BiasInitialisation.Xavier;
                            break;
                    }
                }
                while (biasInitialisation == null);

                Console.WriteLine("Does this layer connect directly and *only* from the immediately previous layer?");
                string directConnectionReponse = Console.ReadLine();
                if (directConnectionReponse.ToLower() == "y")
                {
                    DenseLayer newLayer = new DenseLayer(units, (Activation)activation,
                    (WeightInitialisation)weightInitialisation, (BiasInitialisation)biasInitialisation, layers.Last());
                    model.AddLayer(layers.Last(), newLayer);
                    layers.Add(newLayer);
                }
                else
                {
                    int singleOrMultipleIndices = ChooseOption(new List<string>() { "One previous layer",
                        "Merges multiple previous layers"}, "Does this layer connect from one previous layer or does it merge outputs from multiple previous layers?");
                    switch (singleOrMultipleIndices)
                    {
                        case 0:
                            // Case for just one previous layer.
                            Console.WriteLine("Enter index of the previous layer.");
                            int index = -1;
                            do
                            {
                                try
                                {
                                    index = int.Parse(Console.ReadLine());
                                    if (!Enumerable.Range(0, layers.Count).ToList().Contains(index))
                                    {
                                        // For input integers which aren't in the correct range.
                                        Console.WriteLine("Invalid number.");
                                    }
                                }
                                catch
                                {
                                    // For inputs which are of invalid type.
                                    Console.WriteLine("Invalid number.");
                                }
                            }
                            while (!Enumerable.Range(0, layers.Count).ToList().Contains(index));
                            DenseLayer newLayer = new DenseLayer(units, (Activation)activation,
                            (WeightInitialisation)weightInitialisation, (BiasInitialisation)biasInitialisation, layers[index]);
                            layers.Add(newLayer);
                            model.AddLayer(layers[index], newLayer);
                            break;
                        case 1:
                            // Case for multiple previous layers.
                            Console.WriteLine("How many previous layers is this layer merging?");
                            int previousLayersNumber = -1;
                            do
                            {
                                try
                                {
                                    previousLayersNumber = int.Parse(Console.ReadLine());
                                    // This is because a layer *can* have just one input layer, but then control flow shouldn't pass to this method.
                                    // One previous layer should be handled as standard in the above code.
                                    if (previousLayersNumber < 2)
                                    {
                                        Console.WriteLine("Invalid number of input layers.");
                                    }
                                }
                                catch
                                {
                                    Console.WriteLine("Invalid input.");
                                }
                            }
                            while (previousLayersNumber < 2);

                            int[] layerIndices = new int[previousLayersNumber];
                            for (int i = 0; i < previousLayersNumber; i++)
                            {
                                Console.WriteLine($"Index of layer {i}:");
                                int layerIndex = -1;
                                do
                                {
                                    try
                                    {
                                        layerIndex = int.Parse(Console.ReadLine());
                                        if (layerIndex < 0)
                                        {
                                            Console.WriteLine("Invalid layer index.");
                                        }
                                    }
                                    catch
                                    {
                                        Console.WriteLine("Invalid input.");
                                    }
                                }
                                while (!Enumerable.Range(0, layers.Count).ToList().Contains(layerIndex));
                                layerIndices[i] = layerIndex;
                            }

                            if (layerIndices.Distinct().Count() == layerIndices.Count())
                            {
                                List<Layer> inputLayers = new List<Layer>();
                                for (int i = 0; i < layerIndices.Count(); i++)
                                {
                                    inputLayers.Add(layers[layerIndices[i]]);
                                }
                                int mergeTypeChoice = ChooseOption(new List<string>() { "Concatenate", "Add " }, "Choose merge type:");

                                if (mergeTypeChoice == 0)
                                {
                                    DenseLayer newMergeLayer = new DenseLayer(units, (Activation)activation,
                                    (WeightInitialisation)weightInitialisation, (BiasInitialisation)biasInitialisation,
                                    previousLayers: inputLayers, mergeType: MergeType.Concatenate);
                                    layers.Add(newMergeLayer);
                                    for (int i = 0; i < layerIndices.Length; i++)
                                    {
                                        model.AddLayer(layers[layerIndices[i]], newMergeLayer);
                                    }
                                }
                                else
                                {
                                    try
                                    {
                                        DenseLayer newMergeLayer = new DenseLayer(units, (Activation)activation,
                                        (WeightInitialisation)weightInitialisation, (BiasInitialisation)biasInitialisation,
                                        previousLayers: inputLayers, mergeType: MergeType.Add);
                                        layers.Add(newMergeLayer);
                                        for (int i = 0; i < layerIndices.Length; i++)
                                        {
                                            model.AddLayer(layers[layerIndices[i]], newMergeLayer);
                                        }
                                    }
                                    catch
                                    {
                                        Console.WriteLine("Invalid merge type selected. Concatenate has been selected instead.");
                                        DenseLayer newMergeLayer = new DenseLayer(units, (Activation)activation,
                                        (WeightInitialisation)weightInitialisation, (BiasInitialisation)biasInitialisation,
                                        previousLayers: inputLayers, mergeType: MergeType.Concatenate);
                                        layers.Add(newMergeLayer);
                                        for (int i = 0; i < layerIndices.Length; i++)
                                        {
                                            model.AddLayer(layers[layerIndices[i]], newMergeLayer);
                                        }
                                    }
                                }
                            }
                            else
                            {
                                Console.WriteLine("Invalid layer configuration. Multiple edges are not permitted between two layers.");
                                Console.WriteLine("Did you input the same index twice?");
                            }
                            break;
                    }
                }
                Console.WriteLine("Add more layers? (Y/N)");
                string response = Console.ReadLine();
                if (response.ToLower() == "y")
                {
                    continue;
                }
                else
                {
                    addMoreLayers = false;
                }
            }
            // Cost function is redundant in this implementation as we are running in inference as a demonstration.
            // Holdover from the Model abstract base class.
            try
            {
                model.Compile(CostFunction.MSE);

                Console.WriteLine($"The input size of this model is {layers[0].GetOutputSize()}");

                float[,] inferenceInput = new float[layers[0].GetOutputSize(), 1];
                bool doInference = true;
                while (doInference == true)
                {
                    for (int i = 0; i < inferenceInput.GetLength(0); i++)
                    {
                        Console.WriteLine($"Enter component {i} of the input vector:");
                        try
                        {
                            float component = float.Parse(Console.ReadLine());
                            inferenceInput[i, 0] = component;
                        }
                        catch
                        {
                            Console.WriteLine("Invalid input. Component has been set to 0.");
                            float component = 0;
                            inferenceInput[i, 0] = component;
                        }
                    }

                    float[,] inferenceOutput = model.ForwardPropagate(inferenceInput);
                    Console.WriteLine("Output vector:");
                    for (int i = 0; i < inferenceOutput.GetLength(0); i++)
                    {
                        Console.WriteLine(inferenceOutput[i, 0]);
                    }

                    Console.WriteLine("Would you like to continue inference on examples? (Y/N)");
                    string inferenceResponse = Console.ReadLine();
                    if (inferenceResponse.ToLower() == "y")
                    {
                        doInference = true;
                    }
                    else
                    {
                        doInference = false;
                    }
                }
            }
            catch
            {
                Console.WriteLine("Multiple outputs exposed in model - unsuitable for inference.");
                Console.WriteLine("Please try again!");
            }
        }
    }
}