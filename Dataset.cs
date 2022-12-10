using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public class Dataset
    {
        string _filepath;
        Tuple<List<float[]>, List<float[]>> _dataset;
        int _index = 0;
        int _datasetSize;

        public Dataset(string filepath, int inputFeatureSize, int outputFeatureSize)
        {
            _filepath = filepath;
            _dataset = Data.ExtractDataset(filepath, inputFeatureSize, outputFeatureSize);
            _datasetSize = _dataset.Item1.Count;
        }

        // Returns a single input-output pair.
        public Tuple<float[], float[]> GetData()
        {
            _index++;
            return new Tuple<float[], float[]>(_dataset.Item1[_index - 1], _dataset.Item2[_index - 1]);
        }

        // Returns multiple input-output pairs packaged as vectors.
        public Tuple<float[,], float[,]> GetData(int batchSize)
        {
            if (batchSize > _datasetSize)
            {
                throw new ArgumentException("Batch size cannot be larger than dataset size");
            }

            List<float[]> inputs = new List<float[]>();
            List<float[]> outputs = new List<float[]>();

            Tuple<float[], float[]> pair;

            for (int i = 0; i < batchSize; i++)
            {
                try
                {
                    pair = GetData();
                    inputs.Add(pair.Item1);
                    outputs.Add(pair.Item2);
                }
                catch
                {
                    // Executed when there is no more data left in the dataset to extract.
                    // This will be a common occurence as it is rare that batch size perfectly divides dataset size.
                    // We do not throw an exception, we simply return what we have.
                    if (inputs.Count == 0 || outputs.Count == 0)
                    {
                        return null;
                    }
                    return new Tuple<float[,], float[,]>(Function.ConcatenateVectorsIntoMatrix(inputs), Function.ConcatenateVectorsIntoMatrix(outputs));
                }
            }
            return new Tuple<float[,], float[,]>(Function.ConcatenateVectorsIntoMatrix(inputs), Function.ConcatenateVectorsIntoMatrix(outputs));
        }

        // Resets database/index pointer so that more samples can be received for further epochs.
        public void ResetDatabase()
        {
            _index = 0;
        }
    }
}
