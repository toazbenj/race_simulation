import math
import json

def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def analyze_requests(block):
    """Process one description block."""
    requests = block["requests"]
    results = block["results"]

    true_requests = [req for req, res in zip(requests, results) if res]
    false_requests = [req for req, res in zip(requests, results) if not res]

    comparisons = [
        (tuple(t), tuple(f), euclidean_distance(t, f))
        for t in true_requests
        for f in false_requests
    ]

    return sorted(comparisons, key=lambda x: x[2])[:20]

def analyze_block(dataset, description):
    """Find and process only the block with the matching description."""
    for block in dataset:
        if block["description"] == description:
            return analyze_requests(block)
    raise ValueError(f"No block found with description: {description}")

# Example usage
if __name__ == "__main__":

    dataset = json.load(open("data/paper_data/vector_collision.json"))

    query_desc = "Is Vector Cost: 1, Metric: <function collision_test at 0x751a0f624c10>, Scenario: inside_edge"
    result = analyze_block(dataset, query_desc)

    print("Success, Fail, Distance")
    for r in result:
        print(r)
