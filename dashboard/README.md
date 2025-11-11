# MorphML Dashboard

Real-time web dashboard for monitoring and managing Neural Architecture Search experiments.

## Features

- 📊 **Real-time Monitoring** - Live experiment progress via WebSocket
- 🎯 **Experiment Management** - Create, start, stop, and delete experiments
- 📈 **Convergence Visualization** - Interactive charts showing fitness over time
- 🏗️ **Architecture Viewer** - Graph visualization of neural architectures
- 🎨 **Modern UI** - Material-UI components with responsive design

## Architecture

```
┌─────────────────────────────────────────┐
│         Frontend (React + TypeScript)    │
│  - Material-UI components                │
│  - Recharts for visualization            │
│  - WebSocket for real-time updates       │
│  - Cytoscape.js for graph rendering      │
└─────────────────────────────────────────┘
              ↓ HTTP/WebSocket
┌─────────────────────────────────────────┐
│         Backend (FastAPI)                │
│  - REST API endpoints                    │
│  - WebSocket server                      │
│  - Experiment execution                  │
│  - In-memory storage (demo)              │
└─────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- Node.js 16+ and npm
- MorphML installed

### Backend Setup

```bash
# Navigate to backend directory
cd dashboard/backend

# Install Python dependencies
pip install fastapi uvicorn websockets

# Or install MorphML with API extras
pip install morphml[api]

# Run backend server
python app.py
# Or with uvicorn
uvicorn app:app --reload --port 8000
```

Backend will be available at: http://localhost:8000

- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Frontend Setup

```bash
# Navigate to frontend directory
cd dashboard/frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend will be available at: http://localhost:3000

## Usage

### 1. Start Backend

```bash
cd dashboard/backend
uvicorn app:app --reload --port 8000
```

### 2. Start Frontend

```bash
cd dashboard/frontend
npm start
```

### 3. Access Dashboard

Open http://localhost:3000 in your browser.

### 4. Create Experiment

1. Click "New Experiment" button
2. Enter experiment name
3. Click "Create"
4. Click "Play" icon to start experiment

### 5. Monitor Progress

- View real-time convergence chart
- See best accuracy updates
- Monitor generation progress

## API Endpoints

### Experiments

- `POST /api/v1/experiments` - Create experiment
- `GET /api/v1/experiments` - List experiments
- `GET /api/v1/experiments/{id}` - Get experiment details
- `POST /api/v1/experiments/{id}/start` - Start experiment
- `POST /api/v1/experiments/{id}/stop` - Stop experiment
- `DELETE /api/v1/experiments/{id}` - Delete experiment

### Architectures

- `GET /api/v1/architectures` - List architectures
- `GET /api/v1/architectures/{id}` - Get architecture details

### WebSocket

- `WS /api/v1/stream/{experiment_id}` - Real-time updates

## Development

### Backend Development

```bash
# Run with auto-reload
uvicorn app:app --reload --port 8000

# Run tests
pytest tests/

# Format code
black app.py
```

### Frontend Development

```bash
# Start dev server
npm start

# Build for production
npm run build

# Run tests
npm test

# Type check
npx tsc --noEmit
```

## Configuration

### Backend Configuration

Edit `dashboard/backend/app.py`:

```python
# CORS origins
allow_origins=["http://localhost:3000", "http://localhost:3001"]

# WebSocket update interval
await asyncio.sleep(1)  # Update every second
```

### Frontend Configuration

Edit `dashboard/frontend/package.json`:

```json
{
  "proxy": "http://localhost:8000"
}
```

Or edit `dashboard/frontend/src/App.tsx`:

```typescript
const API_BASE = 'http://localhost:8000/api/v1';
```

## Production Deployment

### Backend

```bash
# Install production dependencies
pip install fastapi uvicorn[standard] gunicorn

# Run with Gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend

```bash
# Build for production
npm run build

# Serve with nginx or any static file server
# Build output is in: dashboard/frontend/build/
```

### Docker Deployment

```dockerfile
# Backend Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# Frontend Dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Troubleshooting

### Backend Issues

**Port already in use:**
```bash
# Change port
uvicorn app:app --port 8001
```

**CORS errors:**
```python
# Add your frontend URL to CORS origins
allow_origins=["http://localhost:3000", "https://yourdomain.com"]
```

### Frontend Issues

**Module not found:**
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

**WebSocket connection failed:**
- Check backend is running on port 8000
- Check firewall settings
- Verify WebSocket URL in ConvergenceChart.tsx

**TypeScript errors:**
```bash
# Install type definitions
npm install --save-dev @types/react @types/node
```

## Features Roadmap

### Current (v0.1)
- ✅ Experiment CRUD operations
- ✅ Real-time progress monitoring
- ✅ Convergence visualization
- ✅ Basic architecture viewer

### Planned (v0.2)
- [ ] User authentication
- [ ] Database persistence (PostgreSQL)
- [ ] Multiple optimizer support
- [ ] Advanced architecture visualization
- [ ] Experiment comparison
- [ ] Export results

### Future (v1.0)
- [ ] Multi-user support
- [ ] Distributed experiment execution
- [ ] Hyperparameter tuning
- [ ] Model deployment integration
- [ ] Performance analytics
- [ ] Custom dashboard widgets

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Support

- Documentation: https://morphml.readthedocs.io
- Issues: https://github.com/TIVerse/MorphML/issues
- Discussions: https://github.com/TIVerse/MorphML/discussions

## Acknowledgments

- FastAPI for the backend framework
- React and Material-UI for the frontend
- Recharts for visualization
- Cytoscape.js for graph rendering
