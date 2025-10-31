"""Tests for DSL system."""


from morphml.core.dsl import Layer, LayerSpec, SearchSpace, create_cnn_space, create_mlp_space


class TestLayerSpec:
    """Tests for LayerSpec."""

    def test_create_layer_spec(self) -> None:
        """Test layer spec creation."""
        spec = LayerSpec("conv2d", {"filters": [32, 64], "kernel_size": [3, 5]})

        assert spec.operation == "conv2d"
        assert "filters" in spec.param_ranges
        assert spec.param_ranges["filters"] == [32, 64]

    def test_sample_layer(self) -> None:
        """Test sampling a concrete layer from spec."""
        spec = LayerSpec("conv2d", {"filters": [32, 64, 128], "kernel_size": [3, 5]})

        # Sample multiple times
        sampled_nodes = [spec.sample() for _ in range(10)]

        # All should be conv2d
        assert all(node.operation == "conv2d" for node in sampled_nodes)

        # Should have variety in parameters
        filter_values = {node.get_param("filters") for node in sampled_nodes}
        assert len(filter_values) > 1  # Should sample different values

    def test_layer_spec_serialization(self) -> None:
        """Test layer spec to_dict and from_dict."""
        original = LayerSpec("dense", {"units": [128, 256, 512]})

        data = original.to_dict()
        restored = LayerSpec.from_dict(data)

        assert restored.operation == original.operation
        assert restored.param_ranges == original.param_ranges


class TestLayerBuilders:
    """Tests for Layer builder class."""

    def test_conv2d_builder(self) -> None:
        """Test conv2d layer builder."""
        spec = Layer.conv2d(filters=[32, 64], kernel_size=[3, 5])

        assert spec.operation == "conv2d"
        assert 32 in spec.param_ranges["filters"]
        assert 3 in spec.param_ranges["kernel_size"]

    def test_dense_builder(self) -> None:
        """Test dense layer builder."""
        spec = Layer.dense(units=[128, 256, 512])

        assert spec.operation == "dense"
        assert 256 in spec.param_ranges["units"]

    def test_maxpool_builder(self) -> None:
        """Test maxpool layer builder."""
        spec = Layer.maxpool(pool_size=2)

        assert spec.operation == "maxpool"
        assert spec.param_ranges["pool_size"] == [2]

    def test_activation_builders(self) -> None:
        """Test activation layer builders."""
        relu_spec = Layer.relu()
        sigmoid_spec = Layer.sigmoid()
        tanh_spec = Layer.tanh()

        assert relu_spec.operation == "relu"
        assert sigmoid_spec.operation == "sigmoid"
        assert tanh_spec.operation == "tanh"

    def test_dropout_builder(self) -> None:
        """Test dropout layer builder."""
        spec = Layer.dropout(rate=[0.3, 0.5])

        assert spec.operation == "dropout"
        assert 0.3 in spec.param_ranges["rate"]

    def test_batchnorm_builder(self) -> None:
        """Test batchnorm layer builder."""
        spec = Layer.batchnorm()

        assert spec.operation == "batchnorm"

    def test_input_builder(self) -> None:
        """Test input layer builder."""
        spec = Layer.input(shape=(3, 32, 32))

        assert spec.operation == "input"
        assert spec.param_ranges["shape"] == [(3, 32, 32)]

    def test_output_builder(self) -> None:
        """Test output layer builder."""
        spec = Layer.output(units=10, activation="softmax")

        assert spec.operation == "dense"
        assert spec.param_ranges["units"] == [10]
        assert spec.metadata.get("is_output") is True

    def test_custom_builder(self) -> None:
        """Test custom layer builder."""
        spec = Layer.custom("my_operation", {"param1": [1, 2, 3]})

        assert spec.operation == "my_operation"
        assert spec.param_ranges["param1"] == [1, 2, 3]

    def test_builder_with_single_values(self) -> None:
        """Test builders with single values (not lists)."""
        spec = Layer.conv2d(filters=64, kernel_size=3)

        # Should convert to lists internally
        assert spec.param_ranges["filters"] == [64]
        assert spec.param_ranges["kernel_size"] == [3]


class TestSearchSpace:
    """Tests for SearchSpace."""

    def test_create_search_space(self) -> None:
        """Test search space creation."""
        space = SearchSpace(name="test_space")

        assert space.name == "test_space"
        assert len(space.layers) == 0

    def test_add_layer(self) -> None:
        """Test adding layers to search space."""
        space = SearchSpace()

        space.add_layer(Layer.conv2d(filters=64))
        space.add_layer(Layer.relu())

        assert len(space.layers) == 2
        assert space.layers[0].operation == "conv2d"
        assert space.layers[1].operation == "relu"

    def test_add_multiple_layers(self) -> None:
        """Test add_layers method."""
        space = SearchSpace()

        space.add_layers(Layer.conv2d(filters=64), Layer.relu(), Layer.maxpool())

        assert len(space.layers) == 3

    def test_method_chaining(self) -> None:
        """Test method chaining."""
        space = (
            SearchSpace("chained")
            .add_layer(Layer.input(shape=(3, 32, 32)))
            .add_layer(Layer.conv2d(filters=64))
            .add_layer(Layer.output(units=10))
        )

        assert len(space.layers) == 3

    def test_sample_architecture(self) -> None:
        """Test sampling architecture from search space."""
        space = SearchSpace("test")

        space.add_layer(Layer.input(shape=(3, 32, 32)))
        space.add_layer(Layer.conv2d(filters=[32, 64], kernel_size=3))
        space.add_layer(Layer.relu())
        space.add_layer(Layer.dense(units=10))

        # Sample an architecture
        arch = space.sample()

        # Should be a valid graph
        assert arch.is_valid()

        # Should have 4 nodes
        assert len(arch.nodes) == 4

        # Should have 3 edges (sequential)
        assert len(arch.edges) == 3

    def test_sample_multiple_architectures(self) -> None:
        """Test sampling multiple different architectures."""
        space = SearchSpace()

        space.add_layer(Layer.input(shape=(3, 32, 32)))
        space.add_layer(Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5, 7]))
        space.add_layer(Layer.dense(units=[128, 256, 512]))
        space.add_layer(Layer.output(units=10))

        # Sample multiple
        architectures = [space.sample() for _ in range(5)]

        # All should be valid
        assert all(arch.is_valid() for arch in architectures)

        # Should have variety (different hashes)
        hashes = {arch.hash() for arch in architectures}
        assert len(hashes) > 1  # Should generate different architectures

    def test_sample_batch(self) -> None:
        """Test batch sampling."""
        space = SearchSpace()
        space.add_layer(Layer.input(shape=(3, 32, 32)))
        space.add_layer(Layer.conv2d(filters=64))
        space.add_layer(Layer.output(units=10))

        batch = space.sample_batch(batch_size=5)

        assert len(batch) == 5
        assert all(arch.is_valid() for arch in batch)

    def test_add_constraint(self) -> None:
        """Test adding constraints."""
        space = SearchSpace()
        space.add_layer(Layer.input(shape=(3, 32, 32)))
        space.add_layer(Layer.conv2d(filters=64))
        space.add_layer(Layer.output(units=10))

        # Add constraint: max 3 nodes
        def max_nodes(graph):
            return len(graph.nodes) <= 3

        space.add_constraint(max_nodes)

        # Should still sample successfully
        arch = space.sample()
        assert len(arch.nodes) <= 3

    def test_get_complexity(self) -> None:
        """Test complexity estimation."""
        space = SearchSpace()

        space.add_layer(Layer.input(shape=(3, 32, 32)))
        space.add_layer(Layer.conv2d(filters=[32, 64], kernel_size=[3, 5]))  # 2 * 2 = 4
        space.add_layer(Layer.dense(units=[128, 256, 512]))  # 3
        space.add_layer(Layer.output(units=10))

        complexity = space.get_complexity()

        assert complexity["num_layers"] == 4
        assert complexity["total_combinations"] > 1

    def test_search_space_serialization(self) -> None:
        """Test search space to_dict and from_dict."""
        original = SearchSpace("test")
        original.add_layer(Layer.conv2d(filters=[32, 64]))
        original.add_layer(Layer.dense(units=10))

        data = original.to_dict()
        restored = SearchSpace.from_dict(data)

        assert restored.name == original.name
        assert len(restored.layers) == len(original.layers)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_cnn_space(self) -> None:
        """Test CNN space creation."""
        space = create_cnn_space(num_classes=10, input_shape=(3, 32, 32))

        assert space.name == "cnn_space"
        assert len(space.layers) > 0

        # Should be able to sample
        arch = space.sample()
        assert arch.is_valid()

    def test_create_mlp_space(self) -> None:
        """Test MLP space creation."""
        space = create_mlp_space(
            num_classes=10, input_shape=(784,), hidden_layers=2, units_range=[128, 256]
        )

        assert space.name == "mlp_space"
        assert len(space.layers) > 0

        # Should be able to sample
        arch = space.sample()
        assert arch.is_valid()

    def test_cnn_space_with_custom_params(self) -> None:
        """Test CNN space with custom parameters."""
        space = create_cnn_space(
            num_classes=100,
            input_shape=(3, 64, 64),
            conv_filters=[[64, 128], [128, 256]],
            dense_units=[[256, 512]],
        )

        arch = space.sample()
        assert arch.is_valid()


def test_complete_workflow() -> None:
    """Integration test: Complete DSL workflow."""
    # Define search space using DSL
    space = SearchSpace(name="cifar10_search")

    # Build with fluent interface
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        # First conv block
        Layer.conv2d(filters=[32, 64], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.batchnorm(),
        Layer.maxpool(pool_size=2),
        # Second conv block
        Layer.conv2d(filters=[64, 128, 256], kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        # Dense layers
        Layer.dense(units=[128, 256, 512]),
        Layer.relu(),
        Layer.dropout(rate=[0.3, 0.5]),
        # Output
        Layer.output(units=10, activation="softmax"),
    )

    # Check complexity
    complexity = space.get_complexity()
    assert complexity["total_combinations"] > 10  # Should have multiple combinations

    # Sample architectures
    arch1 = space.sample()
    arch2 = space.sample()
    arch3 = space.sample()

    # All should be valid
    assert arch1.is_valid()
    assert arch2.is_valid()
    assert arch3.is_valid()

    # Should have variety
    assert len({arch1.hash(), arch2.hash(), arch3.hash()}) >= 2

    # Test serialization
    space_dict = space.to_dict()
    restored_space = SearchSpace.from_dict(space_dict)
    assert restored_space.name == space.name
    assert len(restored_space.layers) == len(space.layers)

    # Restored space should also work
    arch4 = restored_space.sample()
    assert arch4.is_valid()
