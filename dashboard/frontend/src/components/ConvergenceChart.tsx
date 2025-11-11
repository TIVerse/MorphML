/**
 * Convergence Chart Component
 * 
 * Real-time line chart showing fitness convergence over generations.
 * Uses WebSocket for live updates.
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Box, Typography, CircularProgress } from '@mui/material';

interface ConvergenceChartProps {
  experimentId: string;
}

interface DataPoint {
  generation: number;
  fitness: number;
}

function ConvergenceChart({ experimentId }: ConvergenceChartProps) {
  const [data, setData] = useState<DataPoint[]>([]);
  const [status, setStatus] = useState<string>('connecting');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Create WebSocket connection
    const ws = new WebSocket(
      `ws://localhost:8000/api/v1/stream/${experimentId}`
    );

    ws.onopen = () => {
      console.log('WebSocket connected');
      setStatus('connected');
    };

    ws.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data);
        
        // Update data with history
        if (update.history && Array.isArray(update.history)) {
          const newData = update.history.map((fitness: number, index: number) => ({
            generation: index + 1,
            fitness: fitness,
          }));
          setData(newData);
        }
        
        setStatus(update.status || 'running');
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStatus('error');
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setStatus('disconnected');
    };

    wsRef.current = ws;

    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [experimentId]);

  if (status === 'connecting') {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={400}>
        <CircularProgress />
        <Typography variant="body2" sx={{ ml: 2 }}>
          Connecting...
        </Typography>
      </Box>
    );
  }

  if (status === 'error') {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={400}>
        <Typography variant="body1" color="error">
          Error connecting to experiment stream
        </Typography>
      </Box>
    );
  }

  if (data.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={400}>
        <Typography variant="body2" color="textSecondary">
          Waiting for data...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="generation"
            label={{ value: 'Generation', position: 'insideBottom', offset: -5 }}
          />
          <YAxis
            label={{ value: 'Fitness', angle: -90, position: 'insideLeft' }}
            domain={[0, 1]}
          />
          <Tooltip
            formatter={(value: number) => value.toFixed(4)}
            labelFormatter={(label) => `Generation ${label}`}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="fitness"
            stroke="#8884d8"
            strokeWidth={2}
            dot={false}
            name="Best Fitness"
            animationDuration={300}
          />
        </LineChart>
      </ResponsiveContainer>
      
      <Box mt={2} display="flex" justifyContent="space-between">
        <Typography variant="body2" color="textSecondary">
          Status: {status}
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Data points: {data.length}
        </Typography>
        {data.length > 0 && (
          <Typography variant="body2" color="textSecondary">
            Current: {(data[data.length - 1].fitness * 100).toFixed(2)}%
          </Typography>
        )}
      </Box>
    </Box>
  );
}

export default ConvergenceChart;
