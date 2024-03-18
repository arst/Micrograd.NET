# Micrograd.NET

Micrograd.NET is a .NET port of the [Micrograd](https://github.com/karpathy/micrograd) project, a tiny Autograd engine (with a simple neural net) implemented as a direct port of the original Python code.


## Example usage

Simple backprop example:

```csharp
var a = new Value(-4.0);
var b = new Value(2.0);
var c = a + b;
var d = a * b + b.Pow(3);
c += c + 1;
c += 1 + c + (-a);
d += d * 2 + (b + a).Relu();
d += 3 * d + (b - a).Relu();
var e = c - d;
var f = e.Pow(2);
var g = f / 2.0;
g += 10.0 / f;
Console.WriteLine($"{g.Data}"); // Expected to print: 24.7041
g.Backprop();
Console.WriteLine($"{a.Grad}"); // Expected to print: 138.8338
Console.WriteLine($"{b.Grad}"); // Expected to print: 645.5773
```

Simple Multi Layer Perceptron example:

```csharp
var n = new MLP(3, [4, 4, 1]);

var xs = new List<List<double>>
        {
            new List<double> {2.0, 3.0, -1.0},
            new List<double> {3.0, -1.0, 0.5},
            new List<double> {0.5, 1.0, 1.0},
            new List<double> {1.0, 1.0, -1.0}
        };

var ys = new List<double> { 1.0, -1.0, -1.0, 1.0 };

for (int k = 0; k < 100; k++)
{
    var ypred = xs.SelectMany(x => n.Call(x.Select(val => new Value(val)))).ToList();
    var loss = ys.Zip(ypred, (ygt, yout) => (yout - ygt).Pow(2)).ToArray();
    var l = loss[0];

    foreach (var item in loss.Skip(1))
    {
        l = l + item;
    }

    n.ZeroGrad();
    l.Backward();

    foreach (var param in n.Parameters())
    {
        param.Data += -0.01 * param.Grad;
    }

    Console.WriteLine($"{k}, {l}");
}
```

```angular2html
0, data=4.400560893679812, grad=1
1, data=4.157633791178065, grad=1
2, data=4.074255546231063, grad=1
3, data=3.9992781206363643, grad=1
4, data=3.9289842802492227, grad=1
5, data=3.918909622353073, grad=1
.......
97, data=0.6188388752965518, grad=1
98, data=0.596269499466721, grad=1
99, data=0.5742635939848613, grad=1
```

## Tracing / visualization
You need to install the `GraphViz` package to visualize the computation graph. You can download it from [here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html).

```csharp
var n = new Neuron(2);
var x = new Value[]{ new Value(1.0), new Value(-2.0) };
var y = n.Call(x);
y.Backward();
await GraphTracer.RenderGraphToImage(y, "neuron.png");
```

![2d neuron](neuron.png)