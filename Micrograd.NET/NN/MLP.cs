using System.Collections.Generic;
using System.Linq;

namespace Micrograd.NET.NN
{
    public class MLP : Module
    {
        private readonly List<Layer> layers;

        public MLP(int nin, IReadOnlyCollection<int> nouts)
        {
            var sz = new List<int> { nin };
            sz.AddRange(nouts);
            layers = Enumerable.Range(0, nouts.Count).Select(i => new Layer(sz[i], sz[i + 1], i != nouts.Count - 1))
                .ToList();
        }

        public IEnumerable<Value> Call(IEnumerable<Value> x)
        {
            return layers.Aggregate(x, (current, layer) => layer.Call(current));
        }

        public override IEnumerable<Value> Parameters()
        {
            return layers.SelectMany(layer => layer.Parameters());
        }

        public override string ToString()
        {
            return $"MLP of [{string.Join(", ", layers.Select(layer => layer.ToString()))}]";
        }
    }
}