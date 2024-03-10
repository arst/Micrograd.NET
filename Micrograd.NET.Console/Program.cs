using System;
using Micrograd.NET;

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