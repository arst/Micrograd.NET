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
            return neurons.Select(n => n.Call(x)).ToList();
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