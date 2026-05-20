"""
ONNX export and verification for the fish gate classifier.

The trained MobileNetV3-Small weights (.pt) are exported to ONNX format
so they can be loaded in the browser via onnxruntime-web (WASM backend).

Export details:
  - Input  : float32[1, 3, 224, 224]  named "images"  (matches disease model naming)
  - Output : float32[1, 1]            named "output"
  - The output is a raw logit; apply sigmoid in JS:  isFish = sigmoid(logit) > 0.6
  - Opset 17 is compatible with onnxruntime-web >= 1.17

Verification runs a random-input sanity check using onnxruntime (CPU) to confirm
the exported graph loads and produces the expected output shape before we copy
the file to the web project.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mina.core.constants import GATE_IMAGE_SIZE, GATE_RUNS_DIR
from mina.gate_train import build_gate_model


def export_gate_onnx(
    weights_path: Path,
    output_path: Path | None = None,
    opset: int = 17,
) -> Path:
    """
    Export the trained gate classifier weights to ONNX format.

    Args:
        weights_path: Path to fish_gate_best.pt produced by gate_train.py.
        output_path:  Destination for the .onnx file. Defaults to the same
                      directory as weights_path with name fish_gate.onnx.
        opset:        ONNX opset version (17 recommended for ORT-Web >= 1.17).

    Returns:
        Path to the exported .onnx file.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if output_path is None:
        output_path = weights_path.parent / "fish_gate.onnx"

    print(f"Loading weights from: {weights_path}")
    model = build_gate_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.zeros(1, 3, GATE_IMAGE_SIZE, GATE_IMAGE_SIZE)

    print(f"Exporting to ONNX (opset={opset}) → {output_path}")

    # Patch out the internal onnxscript hook that triggers `import onnx` (and
    # the ml_dtypes version conflict).  _add_onnxscript_fn only injects
    # onnxscript-defined custom ops into the proto — MobileNetV3-Small uses
    # only standard ONNX primitives, so this is a no-op for our model.
    try:
        import torch.onnx._internal.torchscript_exporter.onnx_proto_utils as _pu
        _pu._add_onnxscript_fn = lambda proto, opset_version: proto
    except (ImportError, AttributeError):
        pass  # older torch versions don't have this; harmless to skip

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        # Name 'images' intentionally matches the disease worker's input name
        # so the JS gate worker design stays consistent.
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # Force legacy TorchScript exporter
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Export complete: {output_path}  ({size_mb:.2f} MB)")
    return output_path


def verify_onnx(onnx_path: Path) -> None:
    """
    Sanity-check the exported ONNX model using onnxruntime (CPU).

    Confirms:
    - The model loads without errors.
    - Output shape is (1, 1) as expected by the JS gate worker.
    - Sigmoid(logit) is in a sensible range [0, 1].

    Args:
        onnx_path: Path to the exported fish_gate.onnx.
    """
    import numpy as np
    import onnxruntime as ort

    print(f"\nVerifying: {onnx_path}")
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    # Print input/output metadata for debugging
    for inp in session.get_inputs():
        print(f"  Input  '{inp.name}': shape={inp.shape} dtype={inp.type}")
    for out in session.get_outputs():
        print(f"  Output '{out.name}': shape={out.shape} dtype={out.type}")

    # Run with a random tensor
    dummy = np.random.rand(1, 3, GATE_IMAGE_SIZE, GATE_IMAGE_SIZE).astype(np.float32)
    outputs = session.run(None, {"images": dummy})

    logit = float(outputs[0][0][0])
    prob = 1 / (1 + np.exp(-logit))
    print(f"\n  Random input → logit={logit:.4f}, sigmoid={prob:.4f}")
    print(f"  Output shape: {outputs[0].shape}  ← expected (1, 1)")

    assert outputs[0].shape == (1, 1), (
        f"Unexpected output shape: {outputs[0].shape}. "
        "The JS gate worker expects (1, 1)."
    )
    print("  ✓ Verification passed")
