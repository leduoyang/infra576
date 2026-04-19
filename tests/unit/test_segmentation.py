import pytest
import numpy as np
from src.segmentation import run_segmentation_pipeline
import unittest.mock as mock

def test_run_segmentation_pipeline_sliding_window_structure(tmp_path):
    """
    Verify that the new run_segmentation_pipeline generates strictly
    overlapping 15s chunks instead of variable lengths.
    """
    mock_subclip = mock.MagicMock()
    mock_subclip.audio.iter_chunks.return_value = iter([np.zeros((16000, 2), dtype=np.float32)])
    mock_subclip.fps = 25.0
    mock_subclip.duration = 15.0
    mock_subclip.iter_frames.return_value = iter([])

    mock_clip = mock.MagicMock()
    mock_clip.audio.iter_chunks.return_value = iter([np.zeros((32000, 2), dtype=np.float32)])
    mock_clip.duration = 35.0  # Should generate loops [0, 15], [5, 20], [10, 25], [15, 30], [20, 35], [25, 35], [30, 35]
    mock_clip.subclip.return_value = mock_subclip

    with mock.patch('src.segmentation.VideoFileClip', return_value=mock_clip), \
         mock.patch('src.segmentation.detect_scenes_scenedetect', return_value=[]), \
         mock.patch('src.segmentation._detect_audio_dropouts', return_value=[]), \
         mock.patch('src.segmentation.analyze_audio_from_array',
                    return_value={"audio_energy": 0.0, "spectral_centroid": 0.0, "spectral_bandwidth": 0.0}), \
         mock.patch('src.segmentation.compute_rms_variance_from_array', return_value=0.0), \
         mock.patch('src.segmentation.compute_zcr_from_array', return_value=0.0), \
         mock.patch('src.segmentation.analyze_motion_from_frames', return_value=0.0), \
         mock.patch('src.segmentation.analyze_motion_energy_variance_from_frames', return_value=0.0), \
         mock.patch('src.segmentation.analyze_spatial_edge_density_from_frames', return_value=0.0), \
         mock.patch('src.segmentation.analyze_color_variance_from_frames', return_value=0.0), \
         mock.patch('src.segmentation.analyze_sharpness_from_frames', return_value=0.0), \
         mock.patch('src.segmentation.analyze_hsv_from_frames', return_value=(0.0, 0.0)), \
         mock.patch('src.segmentation.analyze_letterbox_from_frames', return_value=0.0), \
         mock.patch('src.segmentation.analyze_watermark_from_frames', return_value=0.0), \
         mock.patch('src.segmentation.analyze_solid_color_from_frames', return_value=False):

        metadata = {"audio_start_time": 0.0, "duration_seconds": 35.0}
        segments, all_cuts = run_segmentation_pipeline("fake.mp4", metadata)

        # Duration 35 / step 5 = 7 windows
        assert len(segments) == 7
        assert segments[0]["start_seconds"] == 0.0
        assert segments[0]["end_seconds"] == 15.0
        assert segments[1]["start_seconds"] == 5.0
        assert segments[1]["end_seconds"] == 20.0
        assert "pacing_score" in segments[0]
