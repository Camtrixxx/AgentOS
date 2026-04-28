import numpy as np

from control.safety_limiter import SafetyConfig, SafetyLimiter


def test_limiter_clamps_and_limits_delta():
    limiter = SafetyLimiter(
        SafetyConfig(
            joint_lower=np.zeros(3),
            joint_upper=np.ones(3),
            max_delta_per_step=0.1,
        )
    )

    first = limiter.limit(np.array([2.0, -1.0, 0.5]))
    np.testing.assert_allclose(first, np.array([1.0, 0.0, 0.5]))

    second = limiter.limit(np.array([0.0, 1.0, 1.0]))
    np.testing.assert_allclose(second, np.array([0.9, 0.1, 0.6]))

