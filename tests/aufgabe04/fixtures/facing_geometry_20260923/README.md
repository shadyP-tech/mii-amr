# Facing geometry regression frames

Original compressed images from run `stand_explore_exact2_camera_all5_20260923T124047Z`, revision `6c0734c`:

- Blue: candidate 0006, capture 000004 (ambiguous raw head-border alternatives).
- Green: candidate 0005, capture 000023 (workstation verification budget exhaustion).

JSON includes the image digest, original CameraInfo and bounded search inputs. Images are rectified before replay; no QR decoder or synthetic border enters geometry. These fixtures cover acquisition, not replay of the complete ROS timing/association pipeline. OpenCV versions can differ in proposed rails. The local green replay yields a bounded orientation; blue remains ambiguous and must not acquire fabricated precision.

Follow-up: green's local success above omits `ImageSourceSupport.filter()`. Including
the real observer's source-domain filter reproduces its verification-budget
failure on both OpenCV 5.0.0 and ROS 4.5.4: removing invalid canvas rails before
the quota exposes additional valid QR-texture hypotheses. See
`docs/setups/aufgabe04_blue_green_rim_solution_20260923.md` for the corrected
comparison and offline coherent-rim prototype.
