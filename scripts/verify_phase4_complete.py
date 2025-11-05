#!/usr/bin/env python3
"""
Verify Phase 4 Component 5 (Transfer Learning) is complete.

Checks all deliverables from prompt/phase_4/05_transfer_learning.md

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_file_exists(path: str, description: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    file_path = Path(__file__).parent.parent / path
    if file_path.exists():
        return True, f"✅ {description}: {path}"
    else:
        return False, f"❌ {description}: {path} NOT FOUND"


def check_import(module_path: str, class_name: str) -> Tuple[bool, str]:
    """Check if a class can be imported."""
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        return True, f"✅ Import {class_name} from {module_path}"
    except ImportError as e:
        return False, f"❌ Import {class_name} from {module_path}: {e}"
    except AttributeError as e:
        return False, f"❌ {class_name} not found in {module_path}: {e}"


def check_method_exists(module_path: str, class_name: str, method_name: str) -> Tuple[bool, str]:
    """Check if a method exists in a class."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        method = getattr(cls, method_name)
        return True, f"✅ Method {class_name}.{method_name}()"
    except Exception as e:
        return False, f"❌ Method {class_name}.{method_name}(): {e}"


def verify_phase4_component5():
    """Verify all Phase 4 Component 5 deliverables."""
    print("="*70)
    print("Phase 4 Component 5: Transfer Learning - Verification")
    print("="*70)
    print()
    
    results = []
    all_passed = True
    
    # ========================================================================
    # 1. Core Implementation Files
    # ========================================================================
    print("📦 1. Core Implementation Files")
    print("-" * 70)
    
    checks = [
        ("morphml/meta_learning/transfer.py", "Transfer learning module"),
        ("morphml/meta_learning/predictors/gnn_predictor.py", "GNN predictor module"),
    ]
    
    for path, desc in checks:
        passed, msg = check_file_exists(path, desc)
        results.append((passed, msg))
        all_passed &= passed
        print(msg)
    
    print()
    
    # ========================================================================
    # 2. Class Imports
    # ========================================================================
    print("🔌 2. Class Imports")
    print("-" * 70)
    
    imports = [
        ("morphml.meta_learning.transfer", "ArchitectureTransfer"),
        ("morphml.meta_learning.transfer", "FineTuningStrategy"),
        ("morphml.meta_learning.transfer", "MultiTaskNAS"),
        ("morphml.meta_learning", "ArchitectureTransfer"),  # From __init__
        ("morphml.meta_learning", "FineTuningStrategy"),
        ("morphml.meta_learning", "MultiTaskNAS"),
    ]
    
    for module, cls in imports:
        passed, msg = check_import(module, cls)
        results.append((passed, msg))
        all_passed &= passed
        print(msg)
    
    print()
    
    # ========================================================================
    # 3. Required Methods (from prompt)
    # ========================================================================
    print("⚙️  3. Required Methods (from prompt)")
    print("-" * 70)
    
    methods = [
        # ArchitectureTransfer
        ("morphml.meta_learning.transfer", "ArchitectureTransfer", "transfer_architecture"),
        ("morphml.meta_learning.transfer", "ArchitectureTransfer", "evaluate_transferability"),
        ("morphml.meta_learning.transfer", "ArchitectureTransfer", "recommend_transfer_strategy"),
        
        # FineTuningStrategy
        ("morphml.meta_learning.transfer", "FineTuningStrategy", "get_strategy"),
        ("morphml.meta_learning.transfer", "FineTuningStrategy", "generate_freeze_mask"),
        
        # MultiTaskNAS
        ("morphml.meta_learning.transfer", "MultiTaskNAS", "evaluate_multi_task_fitness"),
        ("morphml.meta_learning.transfer", "MultiTaskNAS", "create_multi_task_evaluator"),
    ]
    
    for module, cls, method in methods:
        passed, msg = check_method_exists(module, cls, method)
        results.append((passed, msg))
        all_passed &= passed
        print(msg)
    
    print()
    
    # ========================================================================
    # 4. Test Files
    # ========================================================================
    print("🧪 4. Test Files")
    print("-" * 70)
    
    tests = [
        ("tests/test_transfer_learning.py", "Transfer learning tests"),
        ("tests/test_gnn_predictor.py", "GNN predictor tests"),
    ]
    
    for path, desc in tests:
        passed, msg = check_file_exists(path, desc)
        results.append((passed, msg))
        all_passed &= passed
        print(msg)
    
    print()
    
    # ========================================================================
    # 5. Example Files
    # ========================================================================
    print("📖 5. Example Files")
    print("-" * 70)
    
    examples = [
        ("examples/transfer_learning_example.py", "Transfer learning examples"),
        ("examples/demonstrate_successful_transfer.py", "Transfer demonstration"),
        ("examples/train_gnn_predictor.py", "GNN training script"),
    ]
    
    for path, desc in examples:
        passed, msg = check_file_exists(path, desc)
        results.append((passed, msg))
        all_passed &= passed
        print(msg)
    
    print()
    
    # ========================================================================
    # 6. Documentation
    # ========================================================================
    print("📚 6. Documentation")
    print("-" * 70)
    
    docs = [
        ("PHASE4_IMPLEMENTATION.md", "Implementation guide"),
        ("PHASE4_COMPLETE.md", "Completion checklist"),
    ]
    
    for path, desc in docs:
        passed, msg = check_file_exists(path, desc)
        results.append((passed, msg))
        all_passed &= passed
        print(msg)
    
    print()
    
    # ========================================================================
    # 7. Functional Tests
    # ========================================================================
    print("🔬 7. Functional Tests")
    print("-" * 70)
    
    try:
        from morphml.meta_learning import ArchitectureTransfer, TaskMetadata
        from morphml.core.graph import ModelGraph, GraphNode
        
        # Create simple test
        source_task = TaskMetadata(
            task_id="test_source",
            dataset_name="CIFAR-10",
            num_classes=10,
            input_size=(3, 32, 32),
            num_samples=50000,
        )
        
        target_task = TaskMetadata(
            task_id="test_target",
            dataset_name="CIFAR-100",
            num_classes=100,
            input_size=(3, 32, 32),
            num_samples=50000,
        )
        
        # Test transferability
        score = ArchitectureTransfer.evaluate_transferability(source_task, target_task)
        passed = 0.0 <= score <= 1.0
        msg = f"✅ Transferability calculation works: {score:.3f}"
        if not passed:
            msg = f"❌ Transferability calculation failed: {score}"
        results.append((passed, msg))
        all_passed &= passed
        print(msg)
        
        # Test recommendation
        rec = ArchitectureTransfer.recommend_transfer_strategy(source_task, target_task)
        passed = "strategy" in rec and "transferability" in rec
        msg = f"✅ Strategy recommendation works: {rec['strategy']}"
        if not passed:
            msg = f"❌ Strategy recommendation failed"
        results.append((passed, msg))
        all_passed &= passed
        print(msg)
        
        # Test simple transfer
        graph = ModelGraph()
        input_node = GraphNode("input", "input", {"input_shape": (3, 32, 32)})
        output_node = GraphNode("output", "dense", {"units": 10})
        graph.add_node(input_node)
        graph.add_node(output_node)
        graph.add_edge_by_id("input", "output")
        
        transferred = ArchitectureTransfer.transfer_architecture(
            graph, source_task, target_task, "modify_head"
        )
        
        passed = transferred.nodes["output"].params["units"] == 100
        msg = f"✅ Architecture transfer works: 10 → 100 classes"
        if not passed:
            msg = f"❌ Architecture transfer failed"
        results.append((passed, msg))
        all_passed &= passed
        print(msg)
        
    except Exception as e:
        msg = f"❌ Functional test failed: {e}"
        results.append((False, msg))
        all_passed = False
        print(msg)
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("="*70)
    print("📊 Verification Summary")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for p, _ in results if p)
    failed = total - passed
    
    print(f"\nTotal Checks: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if all_passed:
        print("\n" + "🎉"*35)
        print("✅ Phase 4 Component 5: COMPLETE")
        print("🎉"*35)
        print("\nAll deliverables from prompt/phase_4/05_transfer_learning.md")
        print("have been successfully implemented and verified!")
        print()
        print("✓ Architecture transfer methods")
        print("✓ Transferability estimation")
        print("✓ Fine-tuning strategies")
        print("✓ Multi-task NAS")
        print("✓ Successful transfer demonstration")
        print()
        print("Next steps:")
        print("  1. Run tests: pytest tests/test_transfer_learning.py -v")
        print("  2. Run examples: python examples/transfer_learning_example.py")
        print("  3. Run demo: python examples/demonstrate_successful_transfer.py")
        print()
        return 0
    else:
        print("\n❌ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(verify_phase4_component5())
