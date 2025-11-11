/**
 * MorphML Dashboard - Main Application Component
 * 
 * Provides real-time monitoring and management of NAS experiments.
 */

import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  Button,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Chip,
  Box,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import axios from 'axios';

import ConvergenceChart from './components/ConvergenceChart';
import ArchitectureViewer from './components/ArchitectureViewer';

const API_BASE = 'http://localhost:8000/api/v1';

interface Experiment {
  id: string;
  name: string;
  status: string;
  best_accuracy: number | null;
  generation: number;
  total_generations: number;
  created_at: string;
}

function App() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExp, setSelectedExp] = useState<string | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  // Fetch experiments
  const fetchExperiments = async () => {
    try {
      const response = await axios.get(`${API_BASE}/experiments`);
      setExperiments(response.data);
    } catch (error) {
      console.error('Error fetching experiments:', error);
    }
  };

  useEffect(() => {
    fetchExperiments();
    const interval = setInterval(fetchExperiments, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  // Start experiment
  const handleStart = async (experimentId: string) => {
    try {
      await axios.post(`${API_BASE}/experiments/${experimentId}/start`);
      fetchExperiments();
    } catch (error) {
      console.error('Error starting experiment:', error);
    }
  };

  // Stop experiment
  const handleStop = async (experimentId: string) => {
    try {
      await axios.post(`${API_BASE}/experiments/${experimentId}/stop`);
      fetchExperiments();
    } catch (error) {
      console.error('Error stopping experiment:', error);
    }
  };

  // Delete experiment
  const handleDelete = async (experimentId: string) => {
    if (!window.confirm('Are you sure you want to delete this experiment?')) {
      return;
    }
    
    try {
      await axios.delete(`${API_BASE}/experiments/${experimentId}`);
      fetchExperiments();
      if (selectedExp === experimentId) {
        setSelectedExp(null);
      }
    } catch (error) {
      console.error('Error deleting experiment:', error);
    }
  };

  // Create experiment
  const handleCreate = async (name: string) => {
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/experiments`, {
        name,
        search_space: {
          layers: [
            { type: 'input', shape: [3, 32, 32] },
            { type: 'conv2d', filters: [32, 64], kernel_size: 3 },
            { type: 'relu' },
            { type: 'maxpool', pool_size: 2 },
            { type: 'flatten' },
            { type: 'dense', units: [128, 256] },
            { type: 'dense', units: 10 },
          ],
        },
        optimizer: 'genetic',
        budget: 1000,
        config: {
          population_size: 20,
          num_generations: 50,
        },
      });
      fetchExperiments();
      setCreateDialogOpen(false);
    } catch (error) {
      console.error('Error creating experiment:', error);
    } finally {
      setLoading(false);
    }
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'stopped':
        return 'warning';
      default:
        return 'default';
    }
  };

  // Calculate statistics
  const stats = {
    total: experiments.length,
    running: experiments.filter((e) => e.status === 'running').length,
    completed: experiments.filter((e) => e.status === 'completed').length,
    bestAccuracy: Math.max(
      ...experiments.map((e) => e.best_accuracy || 0),
      0
    ),
  };

  return (
    <div>
      {/* App Bar */}
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            🧬 MorphML Dashboard
          </Typography>
          <IconButton color="inherit" onClick={fetchExperiments}>
            <RefreshIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          {/* Summary Cards */}
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Total Experiments
                </Typography>
                <Typography variant="h4">{stats.total}</Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Running
                </Typography>
                <Typography variant="h4" color="primary">
                  {stats.running}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Completed
                </Typography>
                <Typography variant="h4" color="success.main">
                  {stats.completed}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Best Accuracy
                </Typography>
                <Typography variant="h4">
                  {(stats.bestAccuracy * 100).toFixed(2)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Experiments Table */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box
                  display="flex"
                  justifyContent="space-between"
                  alignItems="center"
                  mb={2}
                >
                  <Typography variant="h6">Experiments</Typography>
                  <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={() => setCreateDialogOpen(true)}
                  >
                    New Experiment
                  </Button>
                </Box>

                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Progress</TableCell>
                      <TableCell>Best Accuracy</TableCell>
                      <TableCell>Created</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {experiments.map((exp) => (
                      <TableRow
                        key={exp.id}
                        hover
                        selected={selectedExp === exp.id}
                        onClick={() => setSelectedExp(exp.id)}
                        sx={{ cursor: 'pointer' }}
                      >
                        <TableCell>{exp.name}</TableCell>
                        <TableCell>
                          <Chip
                            label={exp.status}
                            color={getStatusColor(exp.status)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          {exp.generation}/{exp.total_generations}
                        </TableCell>
                        <TableCell>
                          {exp.best_accuracy
                            ? `${(exp.best_accuracy * 100).toFixed(2)}%`
                            : '-'}
                        </TableCell>
                        <TableCell>
                          {new Date(exp.created_at).toLocaleString()}
                        </TableCell>
                        <TableCell>
                          {exp.status === 'created' && (
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleStart(exp.id);
                              }}
                            >
                              <PlayIcon />
                            </IconButton>
                          )}
                          {exp.status === 'running' && (
                            <IconButton
                              size="small"
                              color="warning"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleStop(exp.id);
                              }}
                            >
                              <StopIcon />
                            </IconButton>
                          )}
                          <IconButton
                            size="small"
                            color="error"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDelete(exp.id);
                            }}
                          >
                            <DeleteIcon />
                          </IconButton>
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
                  <Typography variant="h6" gutterBottom>
                    Convergence Chart
                  </Typography>
                  <ConvergenceChart experimentId={selectedExp} />
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </Container>

      {/* Create Experiment Dialog */}
      <CreateExperimentDialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        onCreate={handleCreate}
        loading={loading}
      />
    </div>
  );
}

// Create Experiment Dialog Component
function CreateExperimentDialog({
  open,
  onClose,
  onCreate,
  loading,
}: {
  open: boolean;
  onClose: () => void;
  onCreate: (name: string) => void;
  loading: boolean;
}) {
  const [name, setName] = useState('');

  const handleSubmit = () => {
    if (name.trim()) {
      onCreate(name);
      setName('');
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Create New Experiment</DialogTitle>
      <DialogContent>
        <TextField
          autoFocus
          margin="dense"
          label="Experiment Name"
          type="text"
          fullWidth
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              handleSubmit();
            }
          }}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={!name.trim() || loading}
        >
          Create
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default App;
