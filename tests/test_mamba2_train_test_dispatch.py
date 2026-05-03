from types import SimpleNamespace

import pytest

from app import h_dqn, mamba2_train_test


def _fake_eval_summary() -> dict:
    return {
        "summary": {
            "avg_return": 1.0,
            "std_return": 0.0,
            "avg_length": 1.0,
            "avg_max_tile": 2.0,
            "avg_illegal_action_attempts": 0.0,
            "total_illegal_action_attempts": 0,
        }
    }


@pytest.mark.parametrize(
    "algorithm",
    [
        "dqn",
        "double_dqn",
        "dueling_double_dqn",
        "qr_dqn",
        "h_dqn",
    ],
)
def test_main_dispatches_all_supported_algorithms(monkeypatch, algorithm):
    args = SimpleNamespace(algorithm=algorithm, output_dir=None)
    calls: list[tuple] = []

    monkeypatch.setattr(mamba2_train_test.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        mamba2_train_test,
        "build_parser",
        lambda: SimpleNamespace(parse_args=lambda: args),
    )

    if algorithm == "h_dqn":
        original_q_network = h_dqn.QNetwork
        monkeypatch.setattr(h_dqn, "QNetwork", original_q_network)

        controller_net = object()
        target_controller_net = object()
        meta_net = object()
        target_meta_net = object()
        q_policy = object()

        def fake_train_h_dqn(parsed_args, device):
            calls.append(("train", parsed_args.algorithm, device.type))
            return (
                controller_net,
                target_controller_net,
                meta_net,
                target_meta_net,
                q_policy,
                [32, 64],
                16,
                4,
            )

        def fake_eval_h_dqn(parsed_args, received_q_policy, num_actions, device):
            calls.append(
                (
                    "evaluate",
                    parsed_args.algorithm,
                    received_q_policy is q_policy,
                    num_actions,
                    device.type,
                )
            )
            return _fake_eval_summary()

        monkeypatch.setattr(h_dqn, "train", fake_train_h_dqn)
        monkeypatch.setattr(h_dqn, "evaluate_policy", fake_eval_h_dqn)
    else:
        q_net = object()
        target_net = object()

        def fake_train_mamba2(parsed_args, device):
            calls.append(("train", parsed_args.algorithm, device.type))
            return q_net, target_net, 16, 4

        def fake_eval_mamba2(parsed_args, received_q_net, num_actions, device):
            is_expected_policy = isinstance(
                received_q_net, mamba2_train_test.ExpectedQPolicy
            )
            calls.append(
                (
                    "evaluate",
                    parsed_args.algorithm,
                    is_expected_policy,
                    num_actions,
                    device.type,
                )
            )
            if algorithm == "qr_dqn":
                assert is_expected_policy
            else:
                assert not is_expected_policy
            return _fake_eval_summary()

        monkeypatch.setattr(mamba2_train_test, "train", fake_train_mamba2)
        monkeypatch.setattr(mamba2_train_test, "evaluate_policy", fake_eval_mamba2)

    mamba2_train_test.main()

    if algorithm == "h_dqn":
        assert calls == [
            ("train", "h_dqn", "cpu"),
            ("evaluate", "h_dqn", True, 4, "cpu"),
        ]
    else:
        expected_evaluate_policy = algorithm == "qr_dqn"
        assert calls == [
            ("train", algorithm, "cpu"),
            ("evaluate", algorithm, expected_evaluate_policy, 4, "cpu"),
        ]
