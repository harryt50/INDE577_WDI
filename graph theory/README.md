# Exploring Global Development Networks: Impact of Zero-Weight Edges

## Project Overview

![Global Development Networks](/images/global_connections.png "Visualization of Trade and Similarity Networks")

Countries are interconnected through trade, investment, and shared development challenges, forming complex global networks. However, missing or negligible connections—represented as **zero-weight edges**—can distort our understanding of these networks. These gaps might indicate missing trade relationships, environmental disparities, or data limitations, potentially skewing analyses of global development patterns.

### Objective
This project leverages the World Bank’s World Development Indicators (WDI) dataset to build two types of networks and investigate the impact of zero-weight edges:
1. A **Trade/Influence Network** (directed graph) based on exports and FDI outflows to analyze influence and connectivity.
2. A **Similarity Network** (undirected graph) based on CO2 emissions and electricity access to identify development communities.
We apply NetworkX algorithms (Louvain, PageRank, Betweenness Centrality, Eigenvector Centrality, Cosine Similarity, and Dijkstra’s Shortest Path) to these networks, focusing on how zero-weight edges affect the results and what they reveal about global development.

### Why It Matters
Zero-weight edges can significantly alter network analysis outcomes, misrepresenting a country’s influence, connectivity, or community structure:
- In the Trade Network, zero-weight edges might indicate missing trade relationships, isolating a country from global markets.
- In the Similarity Network, zero-weight edges might highlight stark differences in CO2 emissions or electricity access, pointing to developmental disparities.
By understanding these effects, we can provide actionable insights for policymakers to address gaps in global cooperation, trade integration, and sustainable development.

## Research Questions
1. How do zero-weight edges affect community detection in the Similarity Network?
2. What is the impact of zero-weight edges on a country’s influence and connectivity in the Trade Network?
3. How do zero-weight edges influence shortest paths in the Trade Network, and what does this mean for global trade efficiency?
4. What patterns do zero-weight edges reveal about global development disparities?

## Methodology

### Data Source
We use the World Bank’s [World Development Indicators (WDI)](https://databank.worldbank.org/source/world-development-indicators) dataset, accessed via the `wbdata` Python library. The selected indicators are:
- **Exports of goods and services (% of GDP) (NE.EXP.GNFS.ZS)**: Proxy for trade relationships in the Trade Network.
- **Foreign direct investment, net outflows (% of GDP) (BM.KLT.DINV.WD.GD.ZS)**: Proxy for investment influence in the Trade Network.
- **CO2 emissions per capita (EN.GHG.ALL.MT.CE.AR5)**: Measures environmental impact for the Similarity Network.
- **Access to electricity (% of population) (EG.ELC.ACCS.ZS)**: Measures infrastructure development for the Similarity Network.
- **GDP per capita (NY.GDP.PCAP.CD)**: Used to select top countries and as a node attribute for context.

### Network Construction
We focus on the top 50 countries by GDP per capita for computational simplicity:
1. **Trade/Influence Network (Directed Graph)**:
   - Nodes: Top 50 countries by GDP.
   - Edges: Weighted by a combined proxy of exports and FDI outflows (normalized difference). Zero-weight edges occur when the difference is large or data is missing, indicating negligible trade/investment relationships.
   - Result: 50 nodes, 1974 edges.
2. **Similarity Network (Undirected Graph)**:
   - Nodes: Top 50 countries by GDP.
   - Edges: Weighted by cosine similarity of CO2 emissions and electricity access. Zero-weight edges occur when similarity is low (e.g., stark developmental disparities).
   - Result: 50 nodes, 882 edges.

### Analysis
We apply the following NetworkX algorithms to both networks:
- **Louvain (Community Detection)**: Identifies communities in the Similarity Network (undirected). Note: Louvain requires the `python-louvain` package.
- **PageRank**: Measures influence in the Trade Network.
- **Betweenness Centrality**: Assesses connectivity in the Trade Network.
- **Eigenvector Centrality**: Evaluates prominence in the Trade Network.
- **Cosine Similarity**: Computes similarity between countries in the Similarity Network.
- **Dijkstra’s Shortest Path**: Finds shortest trade paths in the Trade Network, highlighting connectivity gaps.

### Visualization
We include enhanced visualizations to tell a compelling data story:
- **Network Graphs**: Both networks are visualized with nodes colored by region, node sizes scaled by weighted degree, and edge widths reflecting weights. Convex hulls highlight regional clusters.
- **Pruning**: Edges with weights below a threshold (0.05) are pruned to focus on significant relationships.
- **Labels**: Only the top 8 hubs (by weighted degree) are labeled to avoid clutter.

## Graph Theory Concepts

This project relies on graph theory to model and analyze global development networks. Below are the key concepts used:

### Basics of Graphs
- **Graph**: A mathematical structure used to model relationships, consisting of **nodes** (vertices) and **edges** (links). In this project, nodes represent countries, and edges represent relationships (trade/influence or developmental similarity).
- **Directed vs. Undirected Graphs**:
  - A **directed graph** (digraph) has edges with direction (e.g., A → B), suitable for modeling asymmetric relationships like trade flows. The Trade Network is a directed graph (`nx.DiGraph`) because trade and investment flows are directional.
  - An **undirected graph** has edges without direction (e.g., A — B), suitable for symmetric relationships like similarity. The Similarity Network is an undirected graph (`nx.Graph`) because similarity (e.g., in CO2 emissions) is mutual.
- **Weighted Edges**: Edges can have weights to represent the strength of a relationship. In the Trade Network, edge weights are based on exports and FDI differences; in the Similarity Network, weights are based on cosine similarity of CO2 emissions and electricity access.

### Zero-Weight Edges
- **Definition**: In weighted graphs, edges with a weight of zero (or absent edges) are considered zero-weight edges. They represent negligible or missing relationships.
- **Role in This Project**:
  - In the Trade Network, zero-weight edges occur when the normalized difference in exports or FDI between two countries is large (e.g., >50% for exports), indicating a lack of significant trade or investment influence.
  - In the Similarity Network, zero-weight edges occur when the cosine similarity between two countries’ CO2 emissions and electricity access is negative or below a threshold, indicating developmental disparity.
- **Impact**: Zero-weight edges can isolate nodes, skew centrality metrics (e.g., PageRank, Betweenness), and affect community detection (e.g., Louvain), revealing gaps in global networks.

### Network Metrics and Algorithms
- **Centrality Metrics**:
  - **PageRank**: Measures a node’s influence based on incoming edges, adjusted for edge weights. In the Trade Network, it identifies countries with significant trade influence.
  - **Betweenness Centrality**: Quantifies a node’s role as a connector by measuring how often it lies on the shortest paths between other nodes. In the Trade Network, it highlights trade hubs.
  - **Eigenvector Centrality**: Assesses a node’s importance based on its connections to other important nodes. In the Trade Network, it identifies prominent trade players.
- **Community Detection**:
  - **Louvain Algorithm**: Partitions an undirected graph into communities by maximizing modularity (a measure of how well nodes cluster together). In the Similarity Network, it groups countries with similar CO2 emissions and electricity access.
- **Similarity and Paths**:
  - **Cosine Similarity**: Measures the similarity between two nodes’ edge weight profiles. In the Similarity Network, it compares countries’ developmental similarity.
  - **Dijkstra’s Shortest Path**: Finds the shortest path between two nodes in a weighted graph, using edge weights as distances. In the Trade Network, it reveals efficient trade routes but may be affected by zero-weight edges.

### Relevance to Global Development
Graph theory allows us to model complex relationships in global development:
- The Trade Network captures directional flows of economic influence, where zero-weight edges highlight trade isolation.
- The Similarity Network captures mutual developmental similarities, where zero-weight edges reveal disparities in environmental impact and infrastructure.
By applying graph theory, we can quantify the structural impact of these gaps and translate them into actionable insights for policymakers.

## Key Findings
1. **Developmental Disparities**:
   - In the Similarity Network, zero-weight edges reveal distinct communities based on CO2 emissions and electricity access. Countries with low similarity (e.g., high vs. low emissions) form separate clusters, highlighting environmental and infrastructural disparities.
2. **Trade Isolation**:
   - In the Trade Network, countries with many zero-weight edges (missing trade links) have lower PageRank and Betweenness scores, indicating economic isolation. This suggests potential vulnerabilities for these countries.
3. **Connectivity Gaps**:
   - Zero-weight edges in the Trade Network disrupt shortest paths (Dijkstra’s algorithm), potentially overestimating connectivity and masking real-world trade barriers.
4. **Network Density**:
   - The Trade Network (1974 edges) is denser than the Similarity Network (882 edges), reflecting more trade connections but also more opportunities for zero-weight edges due to missing relationships.

## Policy Implications
- **Address Developmental Gaps**: Countries in different Similarity Network communities need tailored environmental and infrastructure policies to bridge disparities in CO2 emissions and electricity access.
- **Enhance Trade Integration**: Focus on countries with low influence and connectivity in the Trade Network to foster trade relationships, reducing economic isolation.
- **Improve Data Collection**: Zero-weight edges often stem from missing trade or development data. Enhancing data availability can improve network analyses and policy decisions.

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install wbdata pandas numpy networkx matplotlib scipy scikit-learn python-louvain
  ```
  Note: The `python-louvain` package is optional for Louvain community detection. If not installed, the Louvain test will be skipped.

### Running the Code
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Open the Jupyter Notebook:

3. Run all cells to fetch WDI data, build networks, perform the analysis, and visualize results.

### Expected Output
- **Console Output**: A detailed narrative with section headers, explaining the analysis and findings.
- **Visualizations**:
  - **Trade Network Graph**: 50 nodes, 971 edges (after pruning), showing trade relationships with regional clusters.
  - **Similarity Network Graph**: 50 nodes, 877 edges (after pruning), showing developmental similarity with community clusters.
- **Results Table**: A summary of algorithm results and key insights.

## Future Work
- **Use Bilateral Trade Data**: Incorporate actual bilateral trade data (e.g., from UN Comtrade) for more accurate Trade Network edge weights.
- **Expand Scope**: Include all countries, not just the top 50 by GDP, to capture broader patterns.
- **Temporal Analysis**: Analyze how zero-weight edges evolve over time using time-series WDI data.
- **Additional Algorithms**: Test other metrics (e.g., clustering coefficients, label propagation) to further explore network structures.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The [World Bank WDI dataset](https://databank.worldbank.org/source/world-development-indicators) for providing the data.
- NetworkX and the `python-louvain` library for graph analysis tools.
