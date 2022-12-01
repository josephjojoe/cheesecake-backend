using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public abstract class Layer
    {
        public abstract float[] ForwardPass(float[] input);
        
        // Needed for stochastic mini-batch gradient descent.
        public abstract float[,] ForwardPass(float[,] inputs);

        public abstract float[] WeightedOutput(float[] input);

        public abstract float[,] WeightedOutput(float[,] inputs);

        // ForwardPass accepting List<float[]> and List<float[,]> aren't included in this abstract class definition intentionally.
        // This is because InputLayer does not need to implement them - they are, however, implemented in DenseLayer.cs
        public abstract int GetOutputSize();
    }
}
