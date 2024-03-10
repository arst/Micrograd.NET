using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using DotNetGraph.Attributes;
using DotNetGraph.Compilation;
using DotNetGraph.Core;

namespace Micrograd.NET.Trace
{
    public class GraphTracer
    {
        public static async Task RenderGraphToImage(Value root, string outputPath, string rankdir = "LR")
        {
            var (nodes, edges) = Trace(root);
            var graph = new DotGraph
            {
                Directed = true,
                Identifier = new DotIdentifier("Test")
            };

            // Optional: Set graph attributes for layout direction
            if (rankdir == "LR")
                graph.RankDir = new DotRankDirAttribute("LR");
            else
                graph.RankDir = new DotRankDirAttribute("TB");

            var nodeMap = new Dictionary<Value, DotNode>();
            foreach (var node in nodes)
            {
                var dotNode = new DotNode
                {
                    Identifier = new DotIdentifier(node.GetHashCode().ToString()),
                    Label = $"data: {node.Data:0.0000} | grad: {node.Grad:0.0000}"
                };
                graph.Elements.Add(dotNode);
                nodeMap[node] = dotNode;

                if (!string.IsNullOrEmpty(node.Operation))
                {
                    var opNode = new DotNode
                    {
                        Identifier = new DotIdentifier($"{node.GetHashCode()}{node.Operation}"),
                        Label = node.Operation
                    };
                    graph.Elements.Add(opNode);
                    var edgeToOpNode = new DotEdge
                    {
                        From = opNode.Identifier,
                        To = dotNode.Identifier
                    };
                    graph.Elements.Add(edgeToOpNode);
                }
            }

            foreach (var (from, to) in edges)
            {
                var edge = new DotEdge
                {
                    From = nodeMap[from].Identifier,
                    To = new DotIdentifier($"{to.GetHashCode()}{to.Operation}"),
                    ArrowHead = new DotEdgeArrowTypeAttribute(DotEdgeArrowType.Normal),
                    ArrowTail = new DotEdgeArrowTypeAttribute(DotEdgeArrowType.None)
                };
                graph.Elements.Add(edge);
            }

            // Generate DOT language representation
            await using var writer = new StringWriter();
            var context = new CompilationContext(writer, new CompilationOptions());
            await graph.CompileAsync(context);

            var result = writer.GetStringBuilder().ToString();
            await File.WriteAllTextAsync("graph.dot", result);

            // Optionally, use Graphviz to render the DOT file to an image
            // This part assumes you have Graphviz installed and accessible from command line
            var command = $"dot -Tpng graph.dot -o {outputPath}";
            ExecuteCommand(command);
        }

        private static (List<Value> Nodes, List<(Value From, Value To)> Edges) Trace(Value root)
        {
            var nodes = new List<Value>();
            var edges = new List<(Value, Value)>();

            void Build(Value v)
            {
                if (!nodes.Contains(v))
                {
                    nodes.Add(v);
                    foreach (var child in v.Prev)
                    {
                        edges.Add((child, v));
                        Build(child);
                    }
                }
            }

            Build(root);
            return (nodes, edges);
        }

        private static void ExecuteCommand(string command)
        {
            // Execute a shell command
            Process.Start("cmd.exe", $"/c {command}");
        }
    }
}