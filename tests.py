"""Sequential acceptance tests for the xtorch learning project.

The classes are deliberately numbered in implementation order.  Run one stage with

    python -m unittest -v tests.TestStage03TensorArithmetic

or run the complete suite, stopping at the first unfinished behavior, with

    python -m unittest -v -f tests
"""

import unittest

import numpy as np

from xtorch import Tensor, nn, optim
from xtorch.autograd import topological_sort, unbroadcast


def assert_array_close(testcase, actual, expected, *, atol=1e-7, rtol=1e-7):
    """Give more useful failures than comparing NumPy arrays directly."""
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol
    )


class TestStage01TensorBasics(unittest.TestCase):
    def test_construction_properties_and_detach(self):
        source = [[1, 2], [3, 4]]
        x = Tensor(source, requires_grad=True)

        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.data.dtype, np.float64)
        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad)

        detached = x.detach()
        self.assertFalse(detached.requires_grad)
        self.assertFalse(np.shares_memory(detached.data, x.data))
        x.data[0, 0] = 99
        self.assertEqual(detached.data[0, 0], 1)

    def test_zero_grad(self):
        x = Tensor(2, requires_grad=True)
        x.grad = np.array(3.0)
        x.zero_grad()
        self.assertIsNone(x.grad)


class TestStage02AutogradUtilities(unittest.TestCase):
    def test_topological_sort_is_parent_first_and_deduplicated(self):
        a = Tensor(1, requires_grad=True)
        b = Tensor(2, requires_grad=True)
        shared = Tensor(3, _parents=(a, b), _op="shared")
        left = Tensor(4, _parents=(shared,), _op="left")
        right = Tensor(5, _parents=(shared, b), _op="right")
        root = Tensor(6, _parents=(left, right), _op="root")

        order = topological_sort(root)
        self.assertEqual(len(order), 6)
        self.assertEqual(len({id(node) for node in order}), 6)
        positions = {id(node): i for i, node in enumerate(order)}
        for node in order:
            for parent in node._prev:
                self.assertLess(positions[id(parent)], positions[id(node)])

    def test_unbroadcast_handles_scalar_and_multiple_broadcast_axes(self):
        grad = np.arange(24.0).reshape(2, 3, 4)
        assert_array_close(self, unbroadcast(grad, ()), grad.sum())
        assert_array_close(self, unbroadcast(grad, (1, 3, 1)), grad.sum(axis=(0, 2), keepdims=True))
        assert_array_close(self, unbroadcast(grad, (3, 1)), grad.sum(axis=(0, 2)).reshape(3, 1))
        assert_array_close(self, unbroadcast(grad, grad.shape), grad)


class TestStage03TensorArithmetic(unittest.TestCase):
    def test_forward_arithmetic_and_python_scalars(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        y = Tensor([4, 5, 6])

        assert_array_close(self, (x + y).data, [5, 7, 9])
        assert_array_close(self, (x + 2).data, [3, 4, 5])
        assert_array_close(self, (2 + x).data, [3, 4, 5])
        assert_array_close(self, (x * y).data, [4, 10, 18])
        assert_array_close(self, (3 * x).data, [3, 6, 9])
        assert_array_close(self, (-x).data, [-1, -2, -3])
        assert_array_close(self, (10 - x).data, [9, 8, 7])
        assert_array_close(self, (x**3).data, [1, 8, 27])

    def test_tensor_exponents_and_reflected_power(self):
        base = Tensor([2, 3, 4])
        exponent = Tensor([3, 2, 0.5])

        assert_array_close(self, (base**exponent).data, [8, 9, 2])
        assert_array_close(self, (2**Tensor([1, 2, 3])).data, [2, 4, 8])

    def test_requires_grad_propagates(self):
        constant = Tensor([1, 2])
        variable = Tensor([3, 4], requires_grad=True)
        self.assertFalse((constant + 2).requires_grad)
        self.assertTrue((constant + variable).requires_grad)
        self.assertTrue((variable * 2).requires_grad)

    def test_reductions_forward(self):
        x = Tensor(np.arange(24).reshape(2, 3, 4))
        assert_array_close(self, x.sum().data, 276)
        assert_array_close(self, x.sum(axis=1).data, x.data.sum(axis=1))
        self.assertEqual(x.sum(axis=1, keepdims=True).shape, (2, 1, 4))
        assert_array_close(self, x.mean(axis=(0, 2)).data, x.data.mean(axis=(0, 2)))


class TestStage04ScalarAutograd(unittest.TestCase):
    def test_chain_rule_and_shared_nodes(self):
        x = Tensor(3.0, requires_grad=True)
        shared = x * x
        result = shared + shared * x + 2
        result.backward()

        # x^2 + x^3 + 2 has derivative 2x + 3x^2 = 33 at x=3.
        assert_array_close(self, x.grad, 33.0)

    def test_power_negation_and_subtraction_gradients(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(5.0, requires_grad=True)
        ((x**4) - y - x).backward()
        assert_array_close(self, x.grad, 31.0)
        assert_array_close(self, y.grad, -1.0)

    def test_tensor_exponent_gradients(self):
        base = Tensor(2.0, requires_grad=True)
        exponent = Tensor(3.0, requires_grad=True)
        (base**exponent).backward()

        assert_array_close(self, base.grad, 12.0)
        assert_array_close(self, exponent.grad, 8.0 * np.log(2.0))

    def test_reflected_power_gradient(self):
        exponent = Tensor(3.0, requires_grad=True)
        (2**exponent).backward()

        assert_array_close(self, exponent.grad, 8.0 * np.log(2.0))

    def test_leaf_gradients_accumulate_across_backward_calls(self):
        x = Tensor(2.0, requires_grad=True)
        (x * 3).backward()
        (x * 4).backward()
        assert_array_close(self, x.grad, 7.0)

    def test_backward_on_non_scalar_requires_an_explicit_gradient(self):
        x = Tensor([1, 2], requires_grad=True)
        with self.assertRaises((ValueError, RuntimeError)):
            (x * 2).backward()

        (x * 2).backward(np.array([3.0, 4.0]))
        assert_array_close(self, x.grad, [6, 8])


class TestStage05BroadcastAndReductionAutograd(unittest.TestCase):
    def test_broadcast_gradients_are_reduced_to_operand_shapes(self):
        x = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
        bias = Tensor([10, 20, 30], requires_grad=True)
        scale = Tensor([[2], [3]], requires_grad=True)

        ((x + bias) * scale).sum().backward()
        assert_array_close(self, x.grad, [[2, 2, 2], [3, 3, 3]])
        assert_array_close(self, bias.grad, [5, 5, 5])
        assert_array_close(self, scale.grad, [[63], [72]])

    def test_sum_axis_and_mean_gradients(self):
        x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
        weights = np.array([[1.0], [2.0], [3.0]])
        x.sum(axis=(0, 2), keepdims=False).backward(weights[:, 0])
        assert_array_close(self, x.grad, np.broadcast_to(weights, x.shape))

        y = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
        y.mean(axis=0).sum().backward()
        assert_array_close(self, y.grad, np.full((2, 3), 0.5))

    def test_power_broadcasts_and_reduces_both_gradients(self):
        base_data = np.array([[2.0], [3.0]])
        exponent_data = np.array([1.0, 2.0, 3.0])
        base = Tensor(base_data, requires_grad=True)
        exponent = Tensor(exponent_data, requires_grad=True)

        result = base**exponent
        self.assertEqual(result.shape, (2, 3))
        assert_array_close(self, result.data, base_data**exponent_data)
        result.sum().backward()

        expected_base_grad = (
            exponent_data * base_data ** (exponent_data - 1)
        ).sum(axis=1, keepdims=True)
        expected_exponent_grad = (
            base_data**exponent_data * np.log(base_data)
        ).sum(axis=0)
        assert_array_close(self, base.grad, expected_base_grad)
        assert_array_close(self, exponent.grad, expected_exponent_grad)


class TestStage06MatmulAutograd(unittest.TestCase):
    def test_matrix_multiplication_forward_and_backward(self):
        x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        w_data = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])
        x = Tensor(x_data, requires_grad=True)
        w = Tensor(w_data, requires_grad=True)
        out = x @ w

        assert_array_close(self, out.data, x_data @ w_data)
        upstream = np.array([[1.0, 2.0], [3.0, 4.0]])
        out.backward(upstream)
        assert_array_close(self, x.grad, upstream @ w_data.T)
        assert_array_close(self, w.grad, x_data.T @ upstream)

    def test_vector_by_matrix_used_by_linear_layer(self):
        x_data = np.array([1.0, -2.0, 3.0])
        w_data = np.arange(12.0).reshape(3, 4)
        x = Tensor(x_data, requires_grad=True)
        w = Tensor(w_data, requires_grad=True)
        (x @ w).sum().backward()

        assert_array_close(self, x.grad, np.ones(4) @ w_data.T)
        assert_array_close(self, w.grad, np.outer(x_data, np.ones(4)))


class TestStage07ShapeOperations(unittest.TestCase):
    def test_reshape_accepts_args_or_tuple_and_backpropagates(self):
        x = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
        a = x.reshape(3, 2)
        b = a.reshape((6,))
        self.assertEqual(a.shape, (3, 2))
        self.assertEqual(b.shape, (6,))
        (b * Tensor(np.arange(6))).sum().backward()
        assert_array_close(self, x.grad, np.arange(6).reshape(2, 3))

    def test_transpose_default_and_explicit_axes_backpropagate(self):
        x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
        self.assertEqual(x.T.shape, (4, 3, 2))
        y = x.transpose(1, 2, 0)
        self.assertEqual(y.shape, (3, 4, 2))
        upstream = np.arange(24).reshape(3, 4, 2)
        y.backward(upstream)
        assert_array_close(self, x.grad, upstream.transpose(2, 0, 1))


class TestStage08Activations(unittest.TestCase):
    def test_relu_forward_and_backward(self):
        x = Tensor([-2.0, 0.0, 3.0], requires_grad=True)
        y = x.relu()
        assert_array_close(self, y.data, [0, 0, 3])
        y.sum().backward()
        assert_array_close(self, x.grad, [0, 0, 1])

    def test_exp_log_and_tanh_forward_and_backward(self):
        x_data = np.array([0.5, 1.5, 2.5])

        x = Tensor(x_data, requires_grad=True)
        x.exp().sum().backward()
        assert_array_close(self, x.grad, np.exp(x_data))

        x = Tensor(x_data, requires_grad=True)
        x.log().sum().backward()
        assert_array_close(self, x.grad, 1 / x_data)

        x = Tensor(x_data, requires_grad=True)
        x.tanh().sum().backward()
        assert_array_close(self, x.grad, 1 - np.tanh(x_data) ** 2)

    def test_composed_expression_matches_finite_differences(self):
        values = np.array([-0.4, 0.2, 0.8])
        x = Tensor(values, requires_grad=True)
        ((x.tanh().exp() * (x + 2)).mean()).backward()

        def function(v):
            return np.mean(np.exp(np.tanh(v)) * (v + 2))

        eps = 1e-6
        numerical = np.empty_like(values)
        for i in range(values.size):
            plus, minus = values.copy(), values.copy()
            plus[i] += eps
            minus[i] -= eps
            numerical[i] = (function(plus) - function(minus)) / (2 * eps)
        assert_array_close(self, x.grad, numerical, atol=1e-6, rtol=1e-6)


class TestStage09Modules(unittest.TestCase):
    def test_parameter_discovery_is_recursive_and_deduplicated(self):
        first = nn.Linear(2, 3)
        model = nn.Sequential(first, nn.ReLU(), nn.Sequential(nn.Linear(3, 1)))
        model.extra = {"shared": first.weight}
        params = model.parameters()

        self.assertEqual(len(params), 4)
        self.assertEqual(len({id(p) for p in params}), 4)
        self.assertTrue(all(isinstance(p, nn.Parameter) for p in params))

    def test_linear_activation_sequential_forward_and_backward(self):
        layer = nn.Linear(2, 2)
        layer.weight.data[:] = [[1, -1], [2, 3]]
        layer.bias.data[:] = [0.5, -0.5]
        model = nn.Sequential(layer, nn.Tanh())
        x = Tensor([[1.0, 2.0], [-1.0, 1.0]], requires_grad=True)

        output = model(x)
        expected_pre_activation = x.data @ layer.weight.data + layer.bias.data
        assert_array_close(self, output.data, np.tanh(expected_pre_activation))
        output.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(layer.weight.grad.shape, layer.weight.shape)
        self.assertEqual(layer.bias.grad.shape, layer.bias.shape)

    def test_module_and_optimizer_zero_grad(self):
        model = nn.Linear(2, 1)
        model(Tensor([[1.0, 2.0]])).sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        model.zero_grad()
        self.assertTrue(all(p.grad is None for p in model.parameters()))

        for p in model.parameters():
            p.grad = np.ones_like(p.data)
        optim.SGD(model.parameters()).zero_grad()
        self.assertTrue(all(p.grad is None for p in model.parameters()))


class TestStage10Losses(unittest.TestCase):
    def test_mse_loss_value_and_gradient(self):
        prediction = Tensor([[1.0, 3.0], [2.0, -1.0]], requires_grad=True)
        target = Tensor([[0.0, 1.0], [2.0, 1.0]])
        loss = nn.MSELoss()(prediction, target)
        assert_array_close(self, loss.data, 2.25)
        loss.backward()
        assert_array_close(self, prediction.grad, [[0.5, 1.0], [0.0, -1.0]])

    def test_cross_entropy_is_stable_and_has_correct_gradient(self):
        logits_data = np.array([[1001.0, 1002.0, 1003.0], [1.0, -1.0, 0.0]])
        targets = Tensor([2, 0])
        logits = Tensor(logits_data, requires_grad=True)
        loss = nn.CrossEntropyLoss()(logits, targets)

        shifted = logits_data - logits_data.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        expected_loss = -np.log(probabilities[[0, 1], [2, 0]]).mean()
        assert_array_close(self, loss.data, expected_loss)
        self.assertTrue(np.isfinite(loss.data))

        loss.backward()
        expected_grad = probabilities.copy()
        expected_grad[[0, 1], [2, 0]] -= 1
        expected_grad /= 2
        assert_array_close(self, logits.grad, expected_grad)
        self.assertIsNone(targets.grad)


class TestStage11SGD(unittest.TestCase):
    def test_plain_sgd_updates_parameters_and_skips_missing_gradients(self):
        a = nn.Parameter([1.0, -2.0])
        b = nn.Parameter([5.0])
        a.grad = np.array([0.5, -1.0])
        optimizer = optim.SGD([a, b], lr=0.1)
        optimizer.step()

        assert_array_close(self, a.data, [0.95, -1.9])
        assert_array_close(self, b.data, [5.0])

    def test_sgd_momentum_accumulates_velocity(self):
        p = nn.Parameter([1.0, 2.0])
        optimizer = optim.SGD([p], lr=0.1, momentum=0.9)
        p.grad = np.array([1.0, -2.0])
        optimizer.step()
        assert_array_close(self, p.data, [0.9, 2.2])

        p.grad = np.array([0.5, 0.5])
        optimizer.step()
        # velocity = .9 * [1, -2] + [.5, .5] = [1.4, -1.3]
        assert_array_close(self, p.data, [0.76, 2.33])


class TestStage12Adam(unittest.TestCase):
    def test_adam_matches_two_reference_updates(self):
        p = nn.Parameter([1.0, -1.0])
        optimizer = optim.Adam([p], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        expected = p.data.copy()
        m = np.zeros_like(expected)
        v = np.zeros_like(expected)
        for step, grad in enumerate(
            (np.array([0.2, -0.4]), np.array([0.1, 0.3])), start=1
        ):
            p.grad = grad
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            m_hat = m / (1 - 0.9**step)
            v_hat = v / (1 - 0.999**step)
            expected -= 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
            optimizer.step()
            assert_array_close(self, p.data, expected)

        self.assertEqual(optimizer.t, 2)

    def test_adam_skips_parameter_without_gradient(self):
        untouched = nn.Parameter([4.0])
        changed = nn.Parameter([2.0])
        changed.grad = np.array([1.0])
        optimizer = optim.Adam([untouched, changed], lr=0.1)
        optimizer.step()
        assert_array_close(self, untouched.data, [4.0])
        assert_array_close(self, changed.data, [1.9])


class TestStage13EndToEndTraining(unittest.TestCase):
    def test_small_network_learns_a_regression_problem(self):
        np.random.seed(7)
        x = Tensor(np.linspace(-1, 1, 20).reshape(-1, 1))
        target = Tensor(3 * x.data - 0.5)
        model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.03)

        initial_loss = float(criterion(model(x), target).data)
        for _ in range(300):
            optimizer.zero_grad()
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()

        final_loss = float(criterion(model(x), target).data)
        self.assertLess(final_loss, initial_loss * 0.01)
        self.assertLess(final_loss, 0.02)

        # Run the shorter training check from a fresh, deterministic model.
        np.random.seed(7)
        model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
        optimizer = optim.Adam(model.parameters(), lr=0.03)
        criterion = nn.MSELoss()

        initial_loss = float(criterion(model(x), target).data)
        for _ in range(250):
            optimizer.zero_grad()
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
        final_loss = float(criterion(model(x), target).data)

        self.assertLess(final_loss, initial_loss * 0.02)
        self.assertLess(final_loss, 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
