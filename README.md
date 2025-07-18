# Algorithm Mastery: Building Intuition Through Practical Examples

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
*Hands‑on algorithm projects in Python, C#, and beyond*

---

## Overview

This repository accompanies the **Algorithm Mastery** learning series, a collection of seven progressively challenging projects that help developers internalise classic algorithmic techniques by **building** instead of just reading.  Each project ships in two states:

* **Before** – scaffolding, TODOs, and test suites for learners.
* **After**  – the fully‑implemented reference solution with passing tests.

The curriculum leans on *Grokking Algorithms* (Aditya Bhargava, Manning) and extends it with real‑world scenarios, modern tooling, and clean code practices.

---

## Project Catalogue

| # | Project Title                                            | Core Topics                                                                   |
| - | -------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 1 | Efficient Searching and Sorting Algorithms in Action     | Linear search, binary search, selection sort, performance benchmarking        |
| 2 | Dynamic and Recursive Algorithms for Intelligent Systems | Recursion, dynamic programming, greedy heuristics, chatbot context management |
| 3 | Smart Financial Planning with Dynamic Programming        | Knapsack variants, goal‑aware budget allocation, optimisation strategies      |
| 4 | Real‑World Problem Solving with Greedy Algorithms        | Activity selection, interval scheduling, coin change, complexity analysis     |
| 5 | Advanced Graph Traversal and Optimisation Techniques     | BFS/DFS, Dijkstra, A\*, graph modelling, route optimisation                   |
| 6 | Building an AI‑Enhanced Stock Trading System             | Time‑series preprocessing, predictive modelling, risk metrics, back‑testing   |
| 7 | AI‑Powered Recommendation and Prediction Systems         | K‑NN, matrix factorisation, text predictors, offline/online evaluation        |

---

## Repository Layout

```
├── project‑1‑efficient‑searching‑and‑sorting‑algorithms‑in‑action/
│   ├── Before/   # learner version
│   └── After/    # solution version
├── project‑2‑dynamic‑and‑recursive‑algorithms‑for‑intelligent‑systems/
│   ├── Before/
│   └── After/
...
└── project‑7‑ai‑powered‑recommendation‑and‑prediction‑systems/
    ├── Before/
    └── After/
```

> **Tip:** Each *Before* folder contains its own `README` with milestone‑level guidance and a `tests/` directory so you can TDD your way to mastery.

---

## Quick Start

1. **Clone** the repo:

   ```bash
   git clone https://github.com/ronnierahman1/manning‑projects‑algorithm‑mastery.git
   cd manning‑projects‑algorithm‑mastery
   ```
2. **Pick a project** and switch to its `Before` directory.
3. Create a fresh Python virtual environment as required.
4. Run the provided test suite:
5. Implement TODOs until all tests pass.

---

## Prerequisites

* Python ≥ 3.9 for all data‑science tasks (see `requirements.txt` inside each project).
* Git 2.30+ with long‑path support enabled on Windows.

---

## Contributing

Pull requests are welcome!  To propose an improvement or report an issue:

1. **Fork** the repository.
2. Create a feature branch: `git checkout -b feature/descriptive‑name`.
3. Commit your changes with clear messages.
4. Open a PR against `main` and fill in the template.

Please follow the code style guidelines (Black for Python) and ensure all tests pass locally.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

* *Grokking Algorithms* by **Aditya Bhargava** – conceptual backbone.
* Manning Publications – inspiration for the liveProject format.
* Contributors and reviewers who help keep the examples up to date and production‑ready.

Happy learning — and may your algorithms always run in *O(log n)*!
