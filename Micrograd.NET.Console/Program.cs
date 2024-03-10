using System;
using Micrograd.NET;
using Micrograd.NET.Trace;

var a = new Value(3.0);
var b = new Value(2.0);

var c = a + b;

await GraphTracer.RenderGraphToImage(c, "test.png");
Console.WriteLine(c);