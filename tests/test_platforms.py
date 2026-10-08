from jms_bridge.platforms import PLATFORMS

NPU_SMI = """\
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc3                 Version: 24.1.rc3                                             |
+-------------------------------+-----------------+----------------------------------------------+
| NPU   Name                    | Health          | Power(W)   Temp(C)          Hugepages-Usage  |
| Chip  Device                  | Bus-Id          | AICore(%)  Memory-Usage(MB)                  |
+===============================+=================+==============================================+
| 0     910C                    | OK              | 92.3       45               0 / 0           |
| 0     0                       | 0000:01:00.0    | 0          0 / 65536                        |
+-------------------------------+-----------------+----------------------------------------------+
| 1     910C                    | OK              | 88.1       44               0 / 0           |
| 1     0                       | 0000:02:00.0    | 0          0 / 65536                        |
+-------------------------------+-----------------+----------------------------------------------+
"""

NVIDIA_SMI = """\
Thu Oct  9 01:00:00 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07    CUDA Version: 12.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  Off |
|  0%   40C    P8    20W / 450W |      4MiB / 24564MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:02:00.0 Off |                  Off |
|  0%   41C    P8    22W / 450W |      4MiB / 24564MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
"""


def test_ascend_parses_version_and_device_count():
    parsed = PLATFORMS["ascend-910c"].parse_probe(NPU_SMI)
    assert parsed["version"] == "24.1.rc3"
    assert parsed["device_count"] == 2


def test_cuda_parses_version_and_device_count():
    parsed = PLATFORMS["cuda"].parse_probe(NVIDIA_SMI)
    assert parsed["version"] == "12.4"
    assert parsed["device_count"] == 2


def test_probe_always_keeps_raw_output():
    parsed = PLATFORMS["generic"].parse_probe("Linux box 6.1.0 aarch64")
    assert "raw" in parsed
    assert parsed["version"] == "6.1.0"


def test_aliases_resolve():
    assert PLATFORMS["ascend"] is PLATFORMS["ascend-910c"]
    assert PLATFORMS["nvidia"] is PLATFORMS["cuda"]
    assert PLATFORMS["arm64"] is PLATFORMS["aarch64"]