using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public abstract class Model
    {
        public abstract void AddLayer(Layer layer);

        public abstract Tuple<float[], List<float[]>> ForwardPropagate(float[] input);

        public abstract Tuple<float[,], List<float[,]>> ForwardPropagate(float[,] input);

        public abstract void Compile(CostFunction costFunction);

        public abstract void Train(string filename, int epochs = 10, float learningRate = 0.1f, int batchSize = 1);

        public abstract int GetInputSize();

        public abstract int GetOutputSize();
    }
}
