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
        // public abstract float[,] ForwardPass(float[,] input)
        // Needed to be implemented for mini-batch stochastic gradient descent.

        public abstract int GetOutputSize();
    }
}
