Autonomous Delivery Agent

Project Description
This project implements an autonomous delivery agent that navigates a 2D grid city to deliver packages efficiently. The agent can handle:
•	Static obstacles (walls)
•	Varying terrain costs (e.g., roads, rough terrain)
•	Dynamic moving obstacles (e.g., vehicles)
The agent uses multiple planning algorithms:
•	Uninformed search: BFS (Breadth-First Search)
•	Cost-based search: Uniform-Cost Search (UCS)
•	Informed search: A* with an admissible heuristic (Manhattan distance)
•	Local search replanning: Simulated Annealing with path mutations
Dynamic replanning is demonstrated by running the agent on a map with moving obstacles, where it detects blockage and replans in real time.
________________________________________

Repository Structure
.
├── autonomous_delivery_agent.py   # Main source code (all planners + simulation)
├── sample_small.map               # Small test map
├── sample_medium.map              # Medium test map
├── sample_large.map               # Large test map
├── sample_dynamic.map             # Dynamic map with moving obstacles
├── README.md                      # This file
├── requirements.md                # Dependencies and runtime requirements
├── report.md                      # Draft report (convert to PDF for submission)
└── demo/                          # Folder for screenshots or short demo video
________________________________________

Requirements
•	Python 3.8+
•	(Optional) matplotlib for plotting
Install matplotlib:
pip install matplotlib
________________________________________

Usage
1. Save sample maps (if not already present)
python autonomous_delivery_agent.py --save-samples
2. Run planners
# BFS on small map
python autonomous_delivery_agent.py --map sample_small.map --planner bfs

# UCS on medium map
python autonomous_delivery_agent.py --map sample_medium.map --planner ucs

# A* on large map
python autonomous_delivery_agent.py --map sample_large.map --planner astar

# Dynamic replanning on dynamic map
python autonomous_delivery_agent.py --map sample_dynamic.map --planner astar --dynamic-log
Each run prints JSON metrics like:
{"planner": "astar", "path_found": true, "path_cost": 27, "nodes_expanded": 120, "time_s": 0.007}
With --dynamic-log, it also produces a file like dynamic_log_astar.json containing replanning events.
3. Optional plotting
Add --show-plot to visualize the path (requires matplotlib).
python autonomous_delivery_agent.py --map sample_small.map --planner astar --show-plot
________________________________________

Deliverables
1.	Source code: autonomous_delivery_agent.py
2.	Test maps: 4 provided map files
3.	Proof-of-concept dynamic replanning: run with --dynamic-log
4.	Report: included as report.md (convert to PDF)
5.	Demo: add screenshots or a short video in demo/