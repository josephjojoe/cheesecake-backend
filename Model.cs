﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Backend
{
    public abstract class Model
    {
        // NOTE: Both Linear and Complex models will implement AddLayer(), but it hasn't been included as an abstract method here.
        // This is because Linear and Complex models will implement the function with different parameters leading to a differing
        // method signature.
        public abstract void AddLayer(Layer layer);

        public abstract float[] ForwardPropagate(float[] input);

        public abstract float[,] ForwardPropagate(float[,] input);

        public abstract void Compile(CostFunction costFunction);

        public abstract void Train(string filename, int epochs = 10, float learningRate = 0.1f, int batchSize = 1);

        public abstract int GetInputSize();

        public abstract int GetOutputSize();
    }
}
