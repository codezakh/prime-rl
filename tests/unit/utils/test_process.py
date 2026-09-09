import pytest

from prime_rl.utils.process import get_physical_gpu_ids


@pytest.mark.parametrize(
    "visible,expected",
    [
        ("4, 5,6,7", [4, 5, 6, 7]),
        (
            "GPU-212c2889-0415-2295-3b10-89c0ea7c17ee,GPU-f225c809-e2de-b466-c7de-3679fca6950a",
            ["GPU-212c2889-0415-2295-3b10-89c0ea7c17ee", "GPU-f225c809-e2de-b466-c7de-3679fca6950a"],
        ),
        ("", []),
    ],
)
def test_launcher_preserves_visible_device_identifiers(monkeypatch, visible, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert get_physical_gpu_ids() == expected
