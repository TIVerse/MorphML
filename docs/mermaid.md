graph TB
    subgraph "User Interface Layer"
        CLI[CLI Interface]
        API[REST API]
        Dashboard[Web Dashboard]
        Notebooks[Jupyter Notebooks]
    end

    subgraph "Orchestration Layer"
        ExpMgr[Experiment Manager]
        TaskScheduler[Task Scheduler]
        Monitor[Progress Monitor]
        ResourceMgr[Resource Manager]
    end

    subgraph "Core Engine Layer"
        DSLParser[DSL Parser]
        DSLCompiler[DSL Compiler]
        SearchSpace[Search Space]
        SearchEngine[Search Engine]
        GraphSystem[Model Graph System]
        Objectives[Objective Functions]
    end

    subgraph "Optimization Layer"
        Evolutionary[Evolutionary Algorithms]
        Bayesian[Bayesian Optimization]
        GradientNAS[Gradient-based NAS]
        RL[Reinforcement Learning]
        MetaLearn[Meta-Learning Engine]
    end

    subgraph "Execution Layer"
        LocalExec[Local Executor]
        DistWorkers[Distributed Workers]
        Evaluator[Model Evaluator]
        GPUManager[GPU Manager]
    end

    subgraph "Storage Layer"
        ResultDB[(Result Database)]
        CheckpointStore[(Checkpoint Store)]
        ArtifactCache[(Artifact Cache)]
        KnowledgeBase[(Knowledge Base)]
    end

    subgraph "Integration Layer"
        SklearnAdapter[Scikit-learn Adapter]
        PyTorchAdapter[PyTorch Adapter]
        TFAdapter[TensorFlow Adapter]
        JAXAdapter[JAX Adapter]
    end

    CLI --> ExpMgr
    API --> ExpMgr
    Dashboard --> ExpMgr
    Notebooks --> ExpMgr

    ExpMgr --> DSLParser
    ExpMgr --> TaskScheduler
    ExpMgr --> Monitor

    DSLParser --> DSLCompiler
    DSLCompiler --> SearchSpace
    DSLCompiler --> SearchEngine

    SearchEngine --> GraphSystem
    SearchEngine --> Objectives

    TaskScheduler --> Evolutionary
    TaskScheduler --> Bayesian
    TaskScheduler --> GradientNAS
    TaskScheduler --> RL

    Evolutionary --> LocalExec
    Bayesian --> LocalExec
    GradientNAS --> DistWorkers
    RL --> DistWorkers

    LocalExec --> Evaluator
    DistWorkers --> Evaluator

    Evaluator --> SklearnAdapter
    Evaluator --> PyTorchAdapter
    Evaluator --> TFAdapter
    Evaluator --> JAXAdapter

    Evaluator --> ResultDB
    Monitor --> ResultDB
    
    TaskScheduler --> CheckpointStore
    LocalExec --> ArtifactCache
    DistWorkers --> ArtifactCache

    MetaLearn --> KnowledgeBase
    MetaLearn --> Evolutionary
    MetaLearn --> Bayesian

    ResourceMgr --> GPUManager
    GPUManager --> DistWorkers

    style CLI fill:#e1f5ff
    style API fill:#e1f5ff
    style Dashboard fill:#e1f5ff
    style Notebooks fill:#e1f5ff
    
    style ExpMgr fill:#fff4e1
    style TaskScheduler fill:#fff4e1
    style Monitor fill:#fff4e1
    style ResourceMgr fill:#fff4e1
    
    style DSLParser fill:#f0e1ff
    style DSLCompiler fill:#f0e1ff
    style SearchSpace fill:#f0e1ff
    style SearchEngine fill:#f0e1ff
    style GraphSystem fill:#f0e1ff
    style Objectives fill:#f0e1ff
    
    style Evolutionary fill:#e1ffe1
    style Bayesian fill:#e1ffe1
    style GradientNAS fill:#e1ffe1
    style RL fill:#e1ffe1
    style MetaLearn fill:#e1ffe1
    
    style LocalExec fill:#ffe1f5
    style DistWorkers fill:#ffe1f5
    style Evaluator fill:#ffe1f5
    style GPUManager fill:#ffe1f5
    
    style ResultDB fill:#ffe1e1
    style CheckpointStore fill:#ffe1e1
    style ArtifactCache fill:#ffe1e1
    style KnowledgeBase fill:#ffe1e1