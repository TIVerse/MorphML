#!/usr/bin/env python3
"""Verify gRPC and Protocol Buffer setup.

This script checks that protobuf files are generated and can be imported correctly.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_protobuf_files() -> bool:
    """Check if protobuf files exist."""
    print("=" * 60)
    print("  Checking Protocol Buffer Files")
    print("=" * 60)
    print()

    proto_dir = PROJECT_ROOT / "morphml" / "distributed" / "proto"
    
    required_files = [
        "worker.proto",
        "worker_pb2.py",
        "worker_pb2_grpc.py",
        "__init__.py",
    ]

    all_present = True
    for filename in required_files:
        filepath = proto_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✓ {filename:25s} ({size:,} bytes)")
        else:
            print(f"✗ {filename:25s} MISSING")
            all_present = False

    print()
    return all_present


def check_grpc_installed() -> bool:
    """Check if gRPC packages are installed."""
    print("=" * 60)
    print("  Checking gRPC Installation")
    print("=" * 60)
    print()

    packages = {
        "grpc": "grpcio",
        "google.protobuf": "protobuf",
    }

    all_installed = True
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            # Get version if possible
            try:
                if module_name == "grpc":
                    import grpc
                    version = grpc.__version__
                elif module_name == "google.protobuf":
                    import google.protobuf
                    version = google.protobuf.__version__
                else:
                    version = "unknown"
                print(f"✓ {package_name:20s} installed (v{version})")
            except:
                print(f"✓ {package_name:20s} installed")
        except ImportError:
            print(f"✗ {package_name:20s} NOT INSTALLED")
            all_installed = False

    print()
    return all_installed


def test_imports() -> bool:
    """Test importing protobuf modules."""
    print("=" * 60)
    print("  Testing Imports")
    print("=" * 60)
    print()

    tests = [
        ("Basic protobuf import", lambda: __import__("morphml.distributed.proto.worker_pb2")),
        ("gRPC stubs import", lambda: __import__("morphml.distributed.proto.worker_pb2_grpc")),
        ("Message classes", lambda: __import__("morphml.distributed.proto", fromlist=["RegisterRequest"])),
        ("Service classes", lambda: __import__("morphml.distributed.proto", fromlist=["MasterServiceStub"])),
    ]

    all_passed = True
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name}")
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            all_passed = False

    print()
    return all_passed


def test_message_creation() -> bool:
    """Test creating protobuf messages."""
    print("=" * 60)
    print("  Testing Message Creation")
    print("=" * 60)
    print()

    try:
        from morphml.distributed.proto import worker_pb2

        # Test RegisterRequest
        request = worker_pb2.RegisterRequest(
            worker_id="test-worker",
            host="localhost",
            port=50052,
            num_gpus=2,
            gpu_ids=[0, 1],
        )
        request.metadata["test_key"] = "test_value"

        print(f"✓ Created RegisterRequest")
        print(f"  - worker_id: {request.worker_id}")
        print(f"  - host: {request.host}")
        print(f"  - port: {request.port}")
        print(f"  - num_gpus: {request.num_gpus}")
        print(f"  - gpu_ids: {list(request.gpu_ids)}")
        print()

        # Test serialization
        serialized = request.SerializeToString()
        print(f"✓ Serialized to {len(serialized)} bytes")

        # Test deserialization
        request2 = worker_pb2.RegisterRequest()
        request2.ParseFromString(serialized)
        print(f"✓ Deserialized successfully")
        print(f"  - worker_id: {request2.worker_id}")
        print()

        return True

    except Exception as e:
        print(f"✗ Message creation failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_grpc_services() -> bool:
    """Test gRPC service definitions."""
    print("=" * 60)
    print("  Testing gRPC Services")
    print("=" * 60)
    print()

    try:
        from morphml.distributed.proto import worker_pb2_grpc

        print(f"✓ MasterServiceStub available")
        print(f"✓ MasterServiceServicer available")
        print(f"✓ WorkerServiceStub available")
        print(f"✓ WorkerServiceServicer available")
        print(f"✓ add_MasterServiceServicer_to_server available")
        print(f"✓ add_WorkerServiceServicer_to_server available")
        print()

        return True

    except Exception as e:
        print(f"✗ gRPC service test failed: {e}")
        print()
        return False


def main() -> int:
    """Run all verification checks."""
    print()
    print("=" * 60)
    print("  MorphML gRPC Setup Verification")
    print("=" * 60)
    print()

    results = {}

    # Run checks
    results["protobuf_files"] = check_protobuf_files()
    results["grpc_installed"] = check_grpc_installed()
    results["imports"] = test_imports()
    results["message_creation"] = test_message_creation()
    results["grpc_services"] = test_grpc_services()

    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()

    passed = sum(results.values())
    total = len(results)

    for check_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status:8s} - {check_name.replace('_', ' ').title()}")

    print()
    print(f"Result: {passed}/{total} checks passed")
    print()

    if passed == total:
        print("✓ All checks passed! gRPC setup is complete.")
        print()
        print("You can now use distributed execution:")
        print("  from morphml.distributed import MasterNode, WorkerNode")
        print()
        return 0
    else:
        print("✗ Some checks failed. Review errors above.")
        print()
        if not results["protobuf_files"]:
            print("To fix: Run 'python scripts/compile_protos.py'")
        if not results["grpc_installed"]:
            print("To fix: Run 'pip install grpcio protobuf'")
            print("   or: poetry install --extras distributed")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
