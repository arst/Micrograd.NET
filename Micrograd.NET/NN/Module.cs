using System.Collections.Generic;
using System.Linq;

namespace Micrograd.NET.NN
{
    public abstract class Module
    {
        public void ZeroGrad()
        {
            foreach (var p in Parameters()) p.Grad = 0;
        }

        public virtual IEnumerable<Value> Parameters()
        {
            return Enumerable.Empty<Value>();
        }
    }
}