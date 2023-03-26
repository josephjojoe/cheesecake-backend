using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public static class Data
    {
        // Returns two float[][] arrays (packaged in a tuple) containing training data sample inputs and expected outputs.
        // As part of the agreed-upon acceptable limitations with my end user, the text file should contain inputs and
        // outputs in the form
        //
        //      input1,input2,input3,...,inputN|output1,output2,output3,...,outputM
        //
        // on each line. They should all also already be in numerical form and
        // normalised if appropriate/necessary to do so.

        public static Tuple<List<float[]>, List<float[]>> ExtractDataset(string filename, int inputFeatureSize, int outputFeatureSize)
        {
            List<float[]> inputs = new List<float[]>();
            List<float[]> outputs = new List<float[]>();

            if (!filename.EndsWith(".txt"))
            {
                filename += ".txt";
            }

            using (StreamReader reader = new StreamReader(filename))
            {
                string line;
                float[] input;
                float[] output;
                string receivedInputs;
                string receivedOutputs;
                while ((line = reader.ReadLine()) != null)
                {
                    // Separates inputs and outputs
                    receivedInputs = line.Split('|')[0];
                    receivedOutputs = line.Split('|')[1];

                    // Splits inputs and outputs into a list of floats
                    input = Array.ConvertAll(receivedInputs.Split(','), element => float.Parse(element));
                    output = Array.ConvertAll(receivedOutputs.Split(','), element => float.Parse(element));

                    // Fail-safe to ensure compatibility with the model
                    if (input.Length != inputFeatureSize || output.Length != outputFeatureSize)
                    {
                        throw new Exception("Dataset inputs/outputs length didn't match specified inputs/outputs length");
                    }

                    try
                    {
                        inputs.Add(input);
                        outputs.Add(output);
                    }
                    catch
                    {
                        break;
                    }
                }
            }
            return new Tuple<List<float[]>, List<float[]>>(inputs, outputs);
        }
    }
}