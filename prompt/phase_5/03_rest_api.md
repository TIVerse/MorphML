# Component 3: REST API & CLI Enhancements

**Duration:** Week 5  
**LOC Target:** ~3,000  
**Dependencies:** Components 1-2

---

## 🎯 Objective

Complete REST API and enhanced CLI:
1. **Full CRUD API** - Manage all resources
2. **Authentication** - JWT-based auth
3. **Rate Limiting** - Prevent abuse
4. **OpenAPI Docs** - Auto-generated documentation
5. **Advanced CLI** - Interactive commands

---

## 📋 Complete API Specification

```yaml
# openapi.yaml
openapi: 3.0.0
info:
  title: MorphML API
  version: 1.0.0
  description: Neural Architecture Search API

paths:
  /api/v1/experiments:
    get:
      summary: List experiments
      responses:
        '200':
          description: Success
    post:
      summary: Create experiment
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                name:
                  type: string
                search_space:
                  type: object
                optimizer:
                  type: string
  
  /api/v1/experiments/{id}:
    get:
      summary: Get experiment
    delete:
      summary: Delete experiment
  
  /api/v1/architectures:
    get:
      summary: Search architectures
      parameters:
        - name: min_accuracy
          in: query
          schema:
            type: number
  
  /api/v1/search-spaces:
    post:
      summary: Define search space
```

Implementation in `api/main.py` (~2,000 LOC) - Complete FastAPI app with all endpoints.

---

## 📋 Enhanced CLI

### 1. `cli/commands/experiment.py` (~500 LOC)

```python
import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
def experiment():
    """Manage experiments."""
    pass

@experiment.command()
@click.option('--name', required=True)
@click.option('--optimizer', default='genetic')
@click.option('--budget', default=500, type=int)
def create(name, optimizer, budget):
    """Create new experiment."""
    # Interactive search space definition
    console.print("[bold]Define Search Space[/bold]")
    
    # ... interactive prompts
    
    console.print(f"[green]Created experiment: {name}[/green]")

@experiment.command()
def list():
    """List all experiments."""
    # Fetch from API
    experiments = api_client.get_experiments()
    
    table = Table(title="Experiments")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Best Accuracy")
    
    for exp in experiments:
        table.add_row(
            str(exp.id),
            exp.name,
            exp.status,
            f"{exp.best_accuracy:.4f}" if exp.best_accuracy else "-"
        )
    
    console.print(table)
```

---

## ✅ Deliverables

- [ ] Complete REST API with all endpoints
- [ ] JWT authentication
- [ ] Rate limiting
- [ ] OpenAPI/Swagger docs
- [ ] Enhanced interactive CLI
- [ ] API client library

---

**Next:** Final components for visualization and documentation
