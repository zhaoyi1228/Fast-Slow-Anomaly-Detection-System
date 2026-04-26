import base64
import ast
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from anomaly_detection_system.edge.detection.jigsaw_service import JigsawDetector
from anomaly_detection_system.edge.detection.models.model import WideBranchNet


def _encode_dummy_image() -> str:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode('.jpg', image)
    assert ok
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def test_widebranchnet_requires_two_dimensional_num_classes():
    with pytest.raises(TypeError):
        WideBranchNet(time_length=7, num_classes=9)


def test_reference_sample7_output_can_be_reshaped_for_diagonal_scoring():
    model = WideBranchNet(time_length=7, num_classes=[49, 7])
    x = torch.rand(2, 3, 7, 64, 64)

    spatial_logits, temporal_logits = model(x)

    assert spatial_logits.shape == (2, 49)
    assert temporal_logits.shape == (2, 7)
    assert spatial_logits.reshape(2, 7, 7).shape == (2, 7, 7)


def test_detector_does_not_return_fake_scores_when_model_unavailable(tmp_path: Path):
    detector = JigsawDetector(checkpoint_path=str(tmp_path / 'missing_checkpoint.pth'), sample_num=7, gpu_id=-1)
    encoded = _encode_dummy_image()

    for _ in range(7):
        result = detector.detect_single(encoded)

    assert detector._model is None
    assert not (
        result.anomaly_score == 0.5
        and result.spatial_score == 0.3
        and result.temporal_score == 0.7
    )


def test_jigsaw_service_avoids_pep585_builtin_generics_for_python38_compatibility():
    source = Path("anomaly_detection_system/edge/detection/jigsaw_service.py").read_text()
    tree = ast.parse(source)

    builtin_generics = {"tuple", "list", "dict", "set"}
    offending_annotations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in builtin_generics:
                offending_annotations.append(node.value.id)

    assert offending_annotations == []