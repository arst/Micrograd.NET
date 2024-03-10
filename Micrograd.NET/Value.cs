using System;
using System.Collections.Generic;

namespace Micrograd.NET
{
    public class Value
    {
        public Value(double data, Value[]? children = null, string op = "")
        {
            Data = data;
            Grad = 0.0;

            Prev = children ?? Array.Empty<Value>();
            Operation = op;
            Backward = () => { };
        }

        public string Operation { get; }

        public double Grad { get; set; }

        public double Data { get; }

        public Action Backward { get; set; }

        public Value[] Prev { get; }

        public static Value operator +(Value a, Value b)
        {
            return OpAddition(a, b);
        }

        public static Value operator +(Value a, int b)
        {
            var val = new Value(b);
            return OpAddition(a, val);
        }

        public static Value operator +(int b, Value a)
        {
            var val = new Value(b);
            return OpAddition(a, val);
        }

        public static Value operator +(Value a, double b)
        {
            var val = new Value(b);
            return OpAddition(a, val);
        }

        public static Value operator +(double b, Value a)
        {
            var val = new Value(b);
            return OpAddition(a, val);
        }

        public static Value operator *(Value a, Value b)
        {
            return OpMultiply(a, b);
        }

        public static Value operator *(Value a, double b)
        {
            var val = new Value(b);
            return OpMultiply(a, val);
        }

        public static Value operator *(Value a, int b)
        {
            var val = new Value(b);
            return OpMultiply(a, val);
        }

        public static Value operator *(double b, Value a)
        {
            var val = new Value(b);
            return OpMultiply(a, val);
        }

        public static Value operator *(int b, Value a)
        {
            var val = new Value(b);
            return OpMultiply(a, val);
        }

        public Value Pow(float power)
        {
            var result = new Value(Math.Pow(Data, power), new[] { this }, $"**{power}");

            result.Backward = () => { Grad += power * Math.Pow(Data, power - 1) * result.Grad; };

            return result;
        }

        public Value Relu()
        {
            var result = new Value(Data < 0 ? 0 : Data, new[] { this }, "ReLU");

            result.Backward = () => { Grad = (result.Data > 0 ? 1 : 0) * result.Grad; };

            return result;
        }

        public void Backprop()
        {
            var topologicalSorting = new List<Value>();
            var visited = new HashSet<Value>();
            TopologicalSort(this);
            Grad = 1;
            topologicalSorting.Reverse();

            foreach (var node in topologicalSorting) node.Backward();

            void TopologicalSort(Value node)
            {
                if (!visited.Contains(node))
                {
                    visited.Add(node);

                    foreach (var child in node.Prev) TopologicalSort(child);

                    topologicalSorting.Add(node);
                }
            }
        }

        public static Value operator -(Value a)
        {
            return a * -1;
        }

        public static Value operator -(Value a, Value b)
        {
            return a + -b;
        }

        public static Value operator -(Value a, double b)
        {
            return a + -b;
        }

        public static Value operator -(double a, Value b)
        {
            return a + -b;
        }

        public static Value operator -(Value a, int b)
        {
            return a + -b;
        }

        public static Value operator -(int a, Value b)
        {
            return a + -b;
        }

        public static Value operator /(Value a, Value b)
        {
            return a * b.Pow(-1);
        }

        public static Value operator /(Value a, double b)
        {
            var val = new Value(b);
            return a * val.Pow(-1);
        }

        public static Value operator /(Value a, int b)
        {
            var val = new Value(b);
            return a * val.Pow(-1);
        }

        public static Value operator /(double b, Value a)
        {
            var val = new Value(b);
            return val * a.Pow(-1);
        }

        public static Value operator /(int b, Value a)
        {
            var val = new Value(b);
            return a * val.Pow(-1);
        }

        public override string ToString()
        {
            return $"data={Data}, grad={Grad}";
        }

        private static Value OpMultiply(Value a, Value b)
        {
            var result = new Value(a.Data * b.Data, new[] { a, b }, "*");

            result.Backward = () =>
            {
                a.Grad += b.Data * result.Grad;
                b.Grad += a.Data * result.Grad;
            };

            return result;
        }

        private static Value OpAddition(Value a, Value b)
        {
            var result = new Value(a.Data + b.Data, new[] { a, b }, "+");
            result.Backward = () =>
            {
                a.Grad += result.Grad;
                b.Grad += result.Grad;
            };

            return result;
        }
    }
}