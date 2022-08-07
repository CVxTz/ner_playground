from nervaluate import Evaluator
from pprint import pprint

true = [
    [{"label": "A", "start": 2, "end": 4}],
    [{"label": "B", "start": 1, "end": 2}, {"label": "C", "start": 3, "end": 4}],
]

prediction = [
    [{"label": "A", "start": 2, "end": 4}],
    [{"label": "B", "start": 1, "end": 3}, {"label": "C", "start": 5, "end": 6}],
]


evaluator = Evaluator(true, prediction, tags=["A", "B", "C"])

# Returns overall metrics and metrics for each tag

results, results_per_tag = evaluator.evaluate()

pprint(results)

pprint(results_per_tag)
