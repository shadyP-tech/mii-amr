"""Diagnostic videos with frame-aligned source pixels and JSON evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping


RECORDING_FILENAMES = {
    "aufgabe04/stand-axis": "annotated.avi",
    "aufgabe04/stand-axis-mask": "color_mask.avi",
    "aufgabe04/stand-axis-edges": "edges.avi",
    "aufgabe04/stand-axis-side-evidence": "side_evidence.avi",
    "aufgabe04/stand-axis-rectangle": "rectangle.avi",
    "aufgabe04/stand-axis-raw-proposal": "raw_proposal.avi",
}
RECORDING_SCHEMA_VERSION = 1


class DebugWindowRecorder:
    """Record displayed windows without treating nominal AVI fps as sensor time.

    Each JSONL row identifies the corresponding frame in every written video.
    Windows missing from a call are skipped, as in the original viewer recorder,
    and have a null frame index. Source PNGs preserve the supplied decoded pixels;
    callers must supply the unannotated image, including for rejected estimates.
    """

    def __init__(
        self,
        cv2,
        output_directory: Path,
        fps: float,
        *,
        recording_filenames: Mapping[str, str] | None = None,
    ) -> None:
        self._cv2 = cv2
        self._output_directory = Path(output_directory)
        self._fps = fps
        self._recording_filenames = dict(
            RECORDING_FILENAMES if recording_filenames is None else recording_filenames
        )
        self._writers = {}
        self._sizes = {}
        self._filenames = {}
        self._video_indices = {}
        self._session_directory: Path | None = None
        self._metadata_stream = None
        self._frame_index = 0

    @property
    def active(self) -> bool:
        return bool(self._writers)

    def _bgr_frame(self, image):
        if image is None or len(image.shape) not in (2, 3):
            raise ValueError("recording requires a non-empty grayscale or BGR image")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("recording requires non-empty window images")
        if len(image.shape) == 2:
            return self._cv2.cvtColor(image, self._cv2.COLOR_GRAY2BGR)
        if image.shape[2] != 3:
            raise ValueError("recording requires grayscale or BGR images")
        return image

    def start(
        self, images: dict[str, object], *, metadata: dict | None = None, source_frame=None
    ) -> None:
        if self.active:
            return
        if not images:
            raise ValueError("no displayed windows are available to record")
        session_name = (
            "recording_"
            + time.strftime("%Y%m%d_%H%M%S")
            + f"_{time.time_ns() % 1_000_000_000:09d}"
        )
        session_directory = self._output_directory / session_name
        session_directory.mkdir(parents=True, exist_ok=False)
        self._session_directory = session_directory
        self._frame_index = 0
        try:
            codec = self._cv2.VideoWriter_fourcc(*"MJPG")
            for window_name, image in images.items():
                frame = self._bgr_frame(image)
                height, width = frame.shape[:2]
                filename = self._recording_filenames.get(
                    window_name, window_name.replace("/", "_") + ".avi"
                )
                writer = self._cv2.VideoWriter(
                    str(session_directory / filename), codec, self._fps, (width, height)
                )
                self._writers[window_name] = writer
                if not writer.isOpened():
                    raise RuntimeError(f"could not open video writer for {window_name}")
                self._sizes[window_name] = (width, height)
                self._filenames[window_name] = filename
                self._video_indices[window_name] = 0
            self._metadata_stream = (session_directory / "metadata.jsonl").open(
                "x", encoding="utf-8"
            )
            self.write(images, metadata=metadata, source_frame=source_frame)
        except Exception:
            self.stop()
            raise

    def _append_metadata(self, record: dict) -> None:
        payload = json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
        self._metadata_stream.write(payload)
        self._metadata_stream.flush()

    def write(
        self, images: dict[str, object], *, metadata: dict | None = None, source_frame=None
    ) -> None:
        if not self.active:
            return
        source_path = None
        record = None
        try:
            frames = {}
            for window_name in self._writers:
                image = images.get(window_name)
                if image is None:
                    continue
                frame = self._bgr_frame(image)
                width, height = self._sizes[window_name]
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = self._cv2.resize(
                        frame, (width, height), interpolation=self._cv2.INTER_NEAREST
                    )
                frames[window_name] = frame
            if not frames:
                return

            source_name = (
                None if source_frame is None else f"source_{self._frame_index:06d}.png"
            )
            record = {
                **({} if metadata is None else metadata),
                "schema_version": RECORDING_SCHEMA_VERSION,
                "frame_index": self._frame_index,
                "nominal_video_fps": self._fps,
                "source_frame": source_name,
                "window_frames": {filename: None for filename in self._filenames.values()},
                "recording_status": "complete",
            }
            # Reject image arrays/non-finite diagnostics before touching any video.
            json.dumps(record, allow_nan=False)
            if source_frame is not None:
                source = self._bgr_frame(source_frame)
                encoded, payload = self._cv2.imencode(".png", source)
                if not encoded or not len(payload):
                    raise RuntimeError("could not encode original recording frame")
                source_path = self._session_directory / source_name
                temporary = source_path.with_suffix(".png.tmp")
                try:
                    temporary.write_bytes(bytes(payload))
                    temporary.replace(source_path)
                finally:
                    temporary.unlink(missing_ok=True)

            for window_name, frame in frames.items():
                writer = self._writers[window_name]
                if not writer.isOpened():
                    raise RuntimeError(f"video writer closed for {window_name}")
                # OpenCV returns None; explicit False from an adapter means failure.
                if writer.write(frame) is False:
                    raise RuntimeError(f"could not write recording frame for {window_name}")
                record["window_frames"][self._filenames[window_name]] = (
                    self._video_indices[window_name]
                )
                self._video_indices[window_name] += 1
            self._append_metadata(record)
            self._frame_index += 1
        except Exception as exc:
            # A video write cannot be rolled back. Preserve an explicit partial row
            # if any writer succeeded; never label that acquisition as complete.
            if record is not None and any(
                index is not None for index in record["window_frames"].values()
            ):
                record["recording_status"] = "write_failed"
                record["recording_error"] = str(exc)
                try:
                    self._append_metadata(record)
                except Exception:
                    pass  # An I/O failure must still close every writer/JSON file.
            elif source_path is not None:
                try:
                    source_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self.stop()
            if isinstance(exc, (RuntimeError, ValueError)):
                raise
            raise RuntimeError(f"could not write recording evidence: {exc}") from exc

    def stop(self) -> None:
        writers, self._writers = self._writers, {}
        stream, self._metadata_stream = self._metadata_stream, None
        self._sizes = {}
        self._filenames = {}
        self._video_indices = {}
        self._session_directory = None
        try:
            for writer in writers.values():
                try:
                    writer.release()
                except Exception:
                    continue  # Release the remaining handles even if one fails.
        finally:
            if stream is not None:
                stream.close()
