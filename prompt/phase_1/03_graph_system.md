# Component 3: Model Graph System

**Duration:** Week 4  
**LOC Target:** ~2,000  
**Dependencies:** Component 1

---

## 🎯 Objective

Implement DAG-based model representation with nodes, edges, mutations, and serialization.

---

## 📋 Files to Create

### 1. `node.py` (~400 LOC)

**`GraphNode` class:**
- Fields: `id` (UUID), `operation` (str), `params` (dict), `predecessors` (list), `successors` (list), `metadata` (dict)
- Methods: `create()`, `add_predecessor()`, `add_successor()`, `get_param()`, `set_param()`, `clone()`, `to_dict()`, `from_dict()`

**Operations to support:**
- Input/output, conv2d/3d, dense, batch_norm, layer_norm, relu/elu/gelu, max_pool/avg_pool, dropout, flatten, concatenate, add

**Validation:**
- Check operation type exists
- Validate required params present

---

### 2. `edge.py` (~200 LOC)

**`GraphEdge` class:**
- Fields: `source_id`, `target_id`, `data_shape` (optional), `edge_type` ('standard'/'residual'/'attention')
- Methods: `is_compatible()`, `to_dict()`, `from_dict()`

---

### 3. `graph.py` (~600 LOC)

**`ModelGraph` class:**
- Fields: `nodes` (dict), `edges` (list), `input_node_id`, `output_node_id`, `metadata`
- Methods:
  - `add_node()`, `remove_node(reconnect=True)` 
  - `add_edge()`, `remove_edge()`
  - `get_node()`, `get_predecessors()`, `get_successors()`
  - `is_valid_dag()` - use NetworkX
  - `topological_sort()` - execution order
  - `validate()` - check DAG, reachability
  - `clone()`, `to_dict()`, `from_dict()`

**Use NetworkX internally:**
```python
def _build_networkx_graph(self) -> nx.DiGraph:
    G = nx.DiGraph()
    for node_id in self.nodes:
        G.add_node(node_id)
    for edge in self.edges:
        G.add_edge(edge.source_id, edge.target_id)
    return G

def is_valid_dag(self) -> bool:
    return nx.is_directed_acyclic_graph(self._build_networkx_graph())
```

---

### 4. `mutations.py` (~400 LOC)

**Base `GraphMutation` ABC:**
- Methods: `mutate(graph) -> ModelGraph`, `is_applicable(graph) -> bool`

**Concrete mutations:**

**`AddNodeMutation`:**
- Select random edge
- Insert new node with random operation
- Split edge: source → new_node → target

**`RemoveNodeMutation`:**
- Remove non-input/output node
- Reconnect predecessors to successors

**`ModifyNodeMutation`:**
- Change hyperparameters of random node
- Multiply/add noise to numeric params

**`RewireEdgeMutation`:**
- Remove random edge
- Add edge to different target
- Ensure DAG property maintained

**`MutationSelector`:**
- Holds list of mutations with probabilities
- Method: `mutate(graph, num_mutations=1) -> ModelGraph`
- Filter applicable mutations, select weighted random

---

### 5. `serialization.py` (~250 LOC)

```python
def save_graph(graph: ModelGraph, path: str, format: str = 'json') -> None:
    """Save graph as JSON or pickle."""
    if format == 'json':
        with open(path, 'w') as f:
            json.dump(graph.to_dict(), f, indent=2)
    elif format == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(graph, f)

def load_graph(path: str, format: str = 'json') -> ModelGraph:
    """Load graph from file."""
    pass
```

---

### 6. `visualization.py` (~150 LOC)

```python
def plot_graph(graph: ModelGraph, output_path: str = None) -> None:
    """Visualize graph with matplotlib/networkx."""
    G = nx.DiGraph()
    for node_id, node in graph.nodes.items():
        G.add_node(node_id, label=node.operation)
    for edge in graph.edges:
        G.add_edge(edge.source_id, edge.target_id)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.savefig(output_path) if output_path else plt.show()
```

---

## 🧪 Tests

**`test_node.py`:**
- Test node creation, cloning
- Test predecessor/successor management
- Test serialization

**`test_graph.py`:**
- Test add/remove nodes and edges
- Test DAG validation (reject cycles)
- Test topological sort
- Test graph cloning

**`test_mutations.py`:**
- Test each mutation type
- Test mutations preserve DAG property (run 100 times)
- Test MutationSelector probability distribution

---

## ✅ Deliverables

- [ ] All 6 files implemented
- [ ] NetworkX integrated for DAG operations
- [ ] Mutations always produce valid DAGs
- [ ] Serialization roundtrip works
- [ ] Test coverage >80%

---

**Next:** `04_search_engine.md`
