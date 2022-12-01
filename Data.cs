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
        public static Tuple<float[][], float[][]> ExtractDataset(string filename, int inputFeatureSize, int outputFeatureSize)
        {
            float[][] inputs = new float[inputFeatureSize][];
            float[][] outputs = new float[outputFeatureSize][];

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
                int i = 0;
                while ((line = reader.ReadLine()) != null)
                {
                    receivedInputs = line.Split('|')[0];
                    receivedOutputs = line.Split('|')[1];

                    input = Array.ConvertAll(receivedInputs.Split(','), element => float.Parse(element));
                    output = Array.ConvertAll(receivedOutputs.Split(','), element => float.Parse(element));

                    if (input.Length != inputFeatureSize || output.Length != outputFeatureSize)
                    {
                        throw new Exception("Dataset inputs/outputs length didn't match specified inputs/outputs length");
                    }

                    inputs[i] = input;
                    outputs[i] = output;
                    i++;
                }
            }
            return new Tuple<float[][], float[][]>(inputs, outputs);
        }
    }
}
