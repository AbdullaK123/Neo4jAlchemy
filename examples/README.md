# Neo4jAlchemy Examples

This directory contains real-world examples demonstrating the power of Neo4jAlchemy's core data structures.

## 🚀 **Quick Start**

```bash
# Install dependencies
cd examples
pip install -r requirements.txt

# Run any example
python social_network_analysis.py
python company_org_chart.py
python supply_chain_analysis.py
```

## 📊 **Available Examples**

### **1. GitHub Repository Network Analysis** 🔥
**File:** `github_network_analysis.py`
**Description:** Analyzes GitHub repositories, contributors, and their relationships using real GitHub data
**Features:**
- Repository dependency analysis
- Contributor collaboration networks
- Language ecosystem mapping
- Influence scoring and community detection

### **2. Company Organizational Analysis**
**File:** `company_org_chart.py`
**Description:** Models and analyzes company organizational structures
**Features:**
- Hierarchical org chart modeling
- Cross-department collaboration analysis
- Communication pattern analysis
- Leadership influence metrics

### **3. Supply Chain Risk Analysis**
**File:** `supply_chain_analysis.py`
**Description:** Models complex supply chain networks with risk analysis
**Features:**
- Multi-tier supplier networks
- Risk propagation analysis
- Single-point-of-failure detection
- Supply chain optimization

### **4. Social Media Network Analysis**
**File:** `social_media_analysis.py`
**Description:** Analyzes social media interactions and influence patterns
**Features:**
- User interaction modeling
- Influence scoring
- Community detection
- Viral content tracking

### **5. Academic Citation Network**
**File:** `academic_citations.py`
**Description:** Models academic papers and citation relationships
**Features:**
- Citation impact analysis
- Research collaboration networks
- Topic clustering
- Emerging field detection

## 🛠️ **Example Structure**

Each example follows this pattern:

```python
#!/usr/bin/env python3
"""
Example: [Name]
Description: [What it does]
Data Source: [Where the data comes from]
"""

from neo4jalchemy.core.graph import Graph, GraphNode, GraphEdge
import pandas as pd
import json

def load_data():
    """Load real-world data"""
    pass

def build_graph(data):
    """Transform data into graph"""
    pass

def analyze_graph(graph):
    """Perform analysis"""
    pass

def export_results(graph, analysis):
    """Export results to various formats"""
    pass

if __name__ == "__main__":
    main()
```

## 📦 **Dependencies**

Common dependencies across examples:
- `pandas` - Data manipulation
- `requests` - API calls for real data
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization
- `networkx` - Comparison with NetworkX
- `plotly` - Interactive visualizations

## 🎯 **Learning Path**

1. **Start with:** `github_network_analysis.py` - Shows core concepts
2. **Then try:** `social_media_analysis.py` - Real-time data patterns
3. **Advanced:** `supply_chain_analysis.py` - Complex business modeling

## 🔧 **Customization**

Each example can be customized by:
- Changing data sources (CSV, API, JSON)
- Modifying analysis parameters
- Adding custom visualizations
- Exporting to different formats

## 💡 **Tips**

- Examples work with the core Neo4jAlchemy classes (no ORM needed)
- All examples include data export to pandas/JSON
- Each example is self-contained and runnable
- Real data sources are documented in each file

Happy graphing! 🎉