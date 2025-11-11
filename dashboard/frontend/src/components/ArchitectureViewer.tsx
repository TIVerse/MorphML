/**
 * Architecture Viewer Component
 * 
 * Visualizes neural architecture as an interactive graph using Cytoscape.js
 */

import React from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import { Box, Typography } from '@mui/material';

interface ArchitectureViewerProps {
  architecture: {
    nodes: Array<{ id: string; operation: string }>;
    edges: Array<{ source: string; target: string }>;
  };
}

function ArchitectureViewer({ architecture }: ArchitectureViewerProps) {
  // Convert architecture to Cytoscape elements
  const elements = [
    // Nodes
    ...architecture.nodes.map((node) => ({
      data: {
        id: node.id,
        label: node.operation,
      },
    })),
    // Edges
    ...architecture.edges.map((edge, index) => ({
      data: {
        id: `edge-${index}`,
        source: edge.source,
        target: edge.target,
      },
    })),
  ];

  // Cytoscape layout configuration
  const layout = {
    name: 'breadthfirst',
    directed: true,
    padding: 10,
    spacingFactor: 1.5,
  };

  // Cytoscape stylesheet
  const stylesheet = [
    {
      selector: 'node',
      style: {
        'background-color': '#0074D9',
        label: 'data(label)',
        color: '#fff',
        'text-valign': 'center',
        'text-halign': 'center',
        width: 60,
        height: 60,
        'font-size': 12,
      },
    },
    {
      selector: 'edge',
      style: {
        width: 2,
        'line-color': '#ccc',
        'target-arrow-color': '#ccc',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
      },
    },
    {
      selector: 'node[operation="input"]',
      style: {
        'background-color': '#2ECC40',
      },
    },
    {
      selector: 'node[operation="output"]',
      style: {
        'background-color': '#FF4136',
      },
    },
  ];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Architecture Graph
      </Typography>
      <Box
        sx={{
          border: '1px solid #ddd',
          borderRadius: 1,
          overflow: 'hidden',
        }}
      >
        <CytoscapeComponent
          elements={elements}
          layout={layout}
          stylesheet={stylesheet}
          style={{ width: '100%', height: '600px' }}
          zoom={1}
          pan={{ x: 0, y: 0 }}
          minZoom={0.5}
          maxZoom={2}
        />
      </Box>
      <Box mt={2}>
        <Typography variant="body2" color="textSecondary">
          Nodes: {architecture.nodes.length} | Edges: {architecture.edges.length}
        </Typography>
      </Box>
    </Box>
  );
}

export default ArchitectureViewer;
