# Component 1: Web Dashboard

**Duration:** Weeks 1-3  
**LOC Target:** ~5,000  
**Dependencies:** Phase 1-4 complete

---

## 🎯 Objective

Build interactive web dashboard:
1. **React Frontend** - Modern UI with Material-UI
2. **FastAPI Backend** - REST API + WebSocket
3. **Real-time Updates** - Live experiment monitoring
4. **Visualizations** - Charts, graphs, architecture diagrams

---

## 📋 Files to Create

### 1. Backend: `dashboard/backend/app.py` (~1,500 LOC)

```python
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(title="MorphML Dashboard API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ExperimentCreate(BaseModel):
    name: str
    search_space: dict
    optimizer: str
    budget: int
    config: Optional[dict] = {}


class ExperimentResponse(BaseModel):
    id: int
    name: str
    status: str
    best_accuracy: Optional[float] = None
    generation: int
    created_at: str


# Database
from morphml.distributed.storage.database import DatabaseManager
db = DatabaseManager('postgresql://localhost/morphml')


# Endpoints
@app.post("/api/v1/experiments", response_model=ExperimentResponse)
async def create_experiment(experiment: ExperimentCreate):
    """Create new experiment."""
    exp_id = db.create_experiment(
        name=experiment.name,
        config={
            'search_space': experiment.search_space,
            'optimizer': experiment.optimizer,
            'budget': experiment.budget
        }
    )
    
    return ExperimentResponse(
        id=exp_id,
        name=experiment.name,
        status='created',
        generation=0,
        created_at=datetime.utcnow().isoformat()
    )


@app.get("/api/v1/experiments", response_model=List[ExperimentResponse])
async def list_experiments():
    """List all experiments."""
    experiments = db.session.query(Experiment).all()
    
    return [
        ExperimentResponse(
            id=exp.id,
            name=exp.name,
            status=exp.status,
            generation=0,  # TODO: get from state
            created_at=exp.created_at.isoformat()
        )
        for exp in experiments
    ]


@app.get("/api/v1/experiments/{experiment_id}")
async def get_experiment(experiment_id: int):
    """Get experiment details."""
    exp = db.session.query(Experiment).get(experiment_id)
    
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Get best architectures
    best_archs = db.get_best_architectures(experiment_id, top_k=10)
    
    return {
        'id': exp.id,
        'name': exp.name,
        'status': exp.status,
        'best_architectures': [
            {
                'id': arch.id,
                'fitness': arch.fitness,
                'metrics': arch.metrics
            }
            for arch in best_archs
        ]
    }


@app.post("/api/v1/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: int):
    """Start experiment execution."""
    # Create background task
    background_tasks.add_task(run_experiment, experiment_id)
    
    return {'status': 'started'}


@app.websocket("/api/v1/stream/{experiment_id}")
async def websocket_endpoint(websocket: WebSocket, experiment_id: int):
    """WebSocket for real-time updates."""
    await websocket.accept()
    
    try:
        while True:
            # Send progress updates
            progress = get_experiment_progress(experiment_id)
            await websocket.send_json(progress)
            await asyncio.sleep(1)
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


async def run_experiment(experiment_id: int):
    """Run experiment in background."""
    # Load experiment config
    exp = db.session.query(Experiment).get(experiment_id)
    
    # Create optimizer
    from morphml.optimizers import get_optimizer
    optimizer = get_optimizer(
        exp.optimizer,
        search_space=SearchSpace(**exp.search_space),
        config=exp.config
    )
    
    # Run
    best = optimizer.optimize()
    
    # Save results
    exp.status = 'completed'
    db.session.commit()
```

---

### 2. Frontend: `dashboard/frontend/src/App.tsx` (~1,000 LOC)

```typescript
import React, { useState, useEffect } from 'react';
import {
  AppBar, Toolbar, Typography, Container, Grid, Card,
  CardContent, Button, Table, TableBody, TableCell,
  TableHead, TableRow
} from '@mui/material';
import { Line } from 'react-chartjs-2';

interface Experiment {
  id: number;
  name: string;
  status: string;
  best_accuracy?: number;
  generation: number;
  created_at: string;
}

function App() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExp, setSelectedExp] = useState<number | null>(null);

  useEffect(() => {
    // Fetch experiments
    fetch('http://localhost:8000/api/v1/experiments')
      .then(res => res.json())
      .then(data => setExperiments(data));
  }, []);

  return (
    <div>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">
            MorphML Dashboard
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" style={{ marginTop: '2rem' }}>
        <Grid container spacing={3}>
          {/* Summary Cards */}
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h5">
                  {experiments.filter(e => e.status === 'running').length}
                </Typography>
                <Typography color="textSecondary">
                  Active Experiments
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Experiment List */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Experiments
                </Typography>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Best Accuracy</TableCell>
                      <TableCell>Generation</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {experiments.map(exp => (
                      <TableRow key={exp.id}>
                        <TableCell>{exp.name}</TableCell>
                        <TableCell>{exp.status}</TableCell>
                        <TableCell>
                          {exp.best_accuracy?.toFixed(4) || '-'}
                        </TableCell>
                        <TableCell>{exp.generation}</TableCell>
                        <TableCell>
                          <Button
                            size="small"
                            onClick={() => setSelectedExp(exp.id)}
                          >
                            View
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </Grid>

          {/* Convergence Chart */}
          {selectedExp && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6">
                    Convergence
                  </Typography>
                  <ConvergenceChart experimentId={selectedExp} />
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </Container>
    </div>
  );
}

function ConvergenceChart({ experimentId }: { experimentId: number }) {
  const [data, setData] = useState<any>({ labels: [], datasets: [] });

  useEffect(() => {
    // WebSocket connection for real-time updates
    const ws = new WebSocket(
      `ws://localhost:8000/api/v1/stream/${experimentId}`
    );

    ws.onmessage = (event) => {
      const progress = JSON.parse(event.data);
      
      setData({
        labels: progress.generations,
        datasets: [{
          label: 'Best Fitness',
          data: progress.best_fitness,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
        }]
      });
    };

    return () => ws.close();
  }, [experimentId]);

  return <Line data={data} />;
}

export default App;
```

---

### 3. `dashboard/frontend/src/components/ArchitectureViewer.tsx` (~500 LOC)

```typescript
import React from 'react';
import Cytoscape from 'cytoscape';
import CytoscapeComponent from 'react-cytoscapejs';

interface ArchitectureViewerProps {
  architecture: any;
}

function ArchitectureViewer({ architecture }: ArchitectureViewerProps) {
  const elements = [
    // Nodes
    ...architecture.nodes.map((node: any) => ({
      data: { id: node.id, label: node.operation }
    })),
    // Edges
    ...architecture.edges.map((edge: any) => ({
      data: {
        source: edge.source,
        target: edge.target
      }
    }))
  ];

  const layout = {
    name: 'breadthfirst',
    directed: true
  };

  return (
    <CytoscapeComponent
      elements={elements}
      layout={layout}
      style={{ width: '100%', height: '600px' }}
    />
  );
}

export default ArchitectureViewer;
```

---

### 4. `dashboard/frontend/package.json` (~50 LOC)

```json
{
  "name": "morphml-dashboard",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@mui/material": "^5.14.0",
    "@emotion/react": "^11.11.0",
    "@emotion/styled": "^11.11.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-chartjs-2": "^5.2.0",
    "chart.js": "^4.3.0",
    "react-cytoscapejs": "^2.0.0",
    "cytoscape": "^3.25.0",
    "axios": "^1.4.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  }
}
```

---

## 🧪 Usage

```bash
# Backend
cd dashboard/backend
uvicorn app:app --reload

# Frontend
cd dashboard/frontend
npm install
npm start

# Access at http://localhost:3000
```

---

## ✅ Deliverables

- [ ] FastAPI backend with REST + WebSocket
- [ ] React frontend with Material-UI
- [ ] Real-time experiment monitoring
- [ ] Interactive architecture visualization
- [ ] Experiment management (create, start, stop)
- [ ] Performance dashboards

---

**Next:** `02_framework_integrations.md`
