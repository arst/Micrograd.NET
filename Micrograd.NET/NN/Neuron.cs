using System;
using System.Collections.Generic;
using System.Linq;

namespace Micrograd.NET.NN
{
    public class Neuron : Module
    {
        private readonly Value b;
        private readonly bool nonlin;
        private readonly List<Value> w;

        public Neuron(int nin, bool nonlin = true)
        {
            var rnd = new Random();
            w = Enumerable.Range(0, nin).Select(_ => new Value(rnd.NextDouble() * 2 - 1)).ToList();
            b = new Value(0);
            this.nonlin = nonlin;
        }

        public Value Call(IEnumerable<Value> x)
        {
            var act = w.Zip(x, (wi, xi) => wi * xi).Aggregate(b, (acc, val) => acc + val);
            return nonlin ? act.Relu() : act;
        }

        public override IEnumerable<Value> Parameters()
        {
            return w.Append(b);
        }

        public override string ToString()
        {
            return $"{(nonlin ? "ReLU" : "Linear")}Neuron({w.Count})";
        }
    }
}