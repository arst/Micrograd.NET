using System.Collections.Generic;
using System.Linq;

namespace Micrograd.NET.NN
{
    public class Layer : Module
    {
        private readonly List<Neuron> neurons;

        public Layer(int nin, int nout, bool nonlin = true)
        {
            neurons = Enumerable.Range(0, nout).Select(_ => new Neuron(nin, nonlin)).ToList();
        }

        public IEnumerable<Value> Call(IEnumerable<Value> x)
        {
            var outList = neurons.Select(n => n.Call(x)).ToList();
            return outList.Count == 1 ? new List<Value> { outList[0] } : outList;
        }

        public override IEnumerable<Value> Parameters()
        {
            return neurons.SelectMany(n => n.Parameters());
        }

        public override string ToString()
        {
            return $"Layer of [{string.Join(", ", neurons.Select(n => n.ToString()))}]";
        }
    }
}