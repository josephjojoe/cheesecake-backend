using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public static class Optimiser
    {
        public static void Fit(LinearModel model, string filename)
        {
            // Collecting dataset which the model is trained upon.
            Tuple<float[][], float[][]> dataset = Data.ExtractDataset(filename, model.GetInputSize(), model.GetOutputSize());

            // Unpacking dataset
            float[][] inputs = dataset.Item1;
            float[][] expectedOutputs = dataset.Item2;

            float[] modelOutput;
            
            foreach (float[] input in inputs)
            {
                modelOutput = model.ForwardPropagate(input);
            }

            // Something something load layers onto a stack so that you can backprop errors and modify weights accordingly.
        }
    }
}
