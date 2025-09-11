
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.support_graph import build_graph, invoke_graph

if __name__ == "__main__":
    graph = build_graph()
    # FAQ example
    print("FAQ:")
    out = invoke_graph(graph, "What is your return policy?")
    print(out, "\n")

    # Order tracking example
    print("Order (no ID):")
    out = invoke_graph(graph, "Where is my order?")
    print(out, "\n")

    # Order with id (example seed orders first)
    from src.tools.orders import seed_example_orders
    seed_example_orders()
    print("Order (with ID):")
    out = invoke_graph(graph, "Track order 12345")
    print(out, "\n")

    # Recommendation
    print("Recommendation:")
    out = invoke_graph(graph, "Can you recommend speakers for outdoor use?")
    print(out, "\n")
