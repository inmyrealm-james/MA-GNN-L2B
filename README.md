# GRAPH CONVOLUTIONAL NETWORK WITH MULTIPLE AGGREGATORS FOR LEARNING TO BRANCH

This repository contains the code and resources for the paper: "GRAPH CONVOLUTIONAL NETWORK WITH MULTIPLE AGGREGATORS FOR LEARNING TO BRANCH" by Nguyen Vinh Toan and Nguyen Van-Hop.

**Authors:**
*   Nguyen Vinh Toan ([nvtoan.iac@gmail.com](mailto:nvtoan.iac@gmail.com))
*   Nguyen Van-Hop ([nvhop@hcmiu.edu.vn](mailto:nvhop@hcmiu.edu.vn))

School of Industrial Engineering and Management,
International University, VNU-HCM,
Quarter 6, Linh Trung Ward, Thu Duc, HoChiMinh City, Vietnam.

## Abstract

This study introduces a Multi-Aggregator Graph Convolutional Network (MAGCN) that enhances variable selection in Mixed-Integer Linear Programming (MILP) solvers by addressing the trade-off between selection accuracy and computational efficiency inherent in traditional branching methods. Unlike prior models that rely on a single aggregation function, the proposed approach uses Principal Neighborhood Aggregation (PNA) to integrate multiple aggregators - mean, sum, max, and standard deviation - along with degree-scalers that adjust neighbor influence dynamically. A fusion Multi-Layer Perceptron (MLP) and skip-connection technique further improve feature propagation and stabilize learning. The model, trained through imitation learning based on strong branching decisions and integrated within a Branch-and-Bound framework, effectively minimizes the number of explored nodes and LP relaxations. Benchmark tests on various MILP instances, including variants of the Multi-Traveling Salesman Problem and Job Shop Scheduling Problem, reveal competitive or superior performance compared to conventional and attention-enhanced GCN architectures in solving complex MILP problems.

## Repository Structure
