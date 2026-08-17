import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class ProductionImportBoundaryTest(unittest.TestCase):
    def test_production_aufgabe04_modules_do_not_import_perception_debug(self):
        scripts_root = ROOT / "scripts" / "aufgabe04"
        offenders = []
        for path in scripts_root.rglob("*.py"):
            if "perception/debug" in path.relative_to(scripts_root).as_posix():
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("scripts.aufgabe04.perception.debug"):
                            offenders.append((path, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.startswith("scripts.aufgabe04.perception.debug"):
                        offenders.append((path, module))
                    if module == "scripts.aufgabe04.perception":
                        for alias in node.names:
                            if alias.name == "debug":
                                offenders.append((path, f"{module}.{alias.name}"))

        self.assertEqual(offenders, [])

    def test_single_station_segment_runner_only_imports_navigation_modules(self):
        runner = ROOT / "scripts" / "aufgabe04" / "navigation" / "run_single_station_segment.py"
        tree = ast.parse(runner.read_text(), filename=str(runner))
        forbidden_prefixes = (
            "scripts.aufgabe04.qr_scanning",
            "scripts.aufgabe04.logistics",
            "scripts.aufgabe04.fleet",
            "scripts.aufgabe04.stations",
            "scripts.aufgabe04.perception",
        )
        offenders = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [node.module or ""]
            else:
                continue

            for module in imported_modules:
                if module.startswith(forbidden_prefixes):
                    offenders.append(module)
                if module.startswith("scripts.aufgabe04.") and not module.startswith(
                    "scripts.aufgabe04.navigation."
                ):
                    offenders.append(module)

        self.assertEqual(offenders, [])

    def test_aufgabe04_does_not_import_aufgabe03(self):
        scripts_root = ROOT / "scripts" / "aufgabe04"
        offenders = []
        for path in scripts_root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported_modules = [node.module or ""]
                else:
                    continue
                for module in imported_modules:
                    if module.startswith("scripts.aufgabe03"):
                        offenders.append((path.relative_to(ROOT), module))

        self.assertEqual(offenders, [])

    def test_qr_scanning_and_task_client_do_not_import_motion_modules(self):
        checked_roots = [
            ROOT / "scripts" / "aufgabe04" / "qr_scanning",
            ROOT / "scripts" / "aufgabe04" / "task_client",
        ]
        always_forbidden_prefixes = (
            "scripts.aufgabe04.navigation",
            "scripts.aufgabe04.fleet",
            "geometry_msgs",
            "nav2_msgs",
        )
        pure_qr_forbidden_prefixes = (
            "scripts.aufgabe04.perception",
            "rclpy",
            "sensor_msgs",
            "cv2",
            "numpy",
        )
        offenders = []
        for checked_root in checked_roots:
            for path in checked_root.rglob("*.py"):
                tree = ast.parse(path.read_text(), filename=str(path))
                rel_path = path.relative_to(ROOT)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        imported_modules = [node.module or ""]
                    else:
                        continue
                    for module in imported_modules:
                        if module.startswith(always_forbidden_prefixes):
                            offenders.append((rel_path, module))
                        if (
                            checked_root.name == "qr_scanning"
                            and path.name != "onboard_camera_node.py"
                            and module.startswith(pure_qr_forbidden_prefixes)
                        ):
                            offenders.append((rel_path, module))
                        if checked_root.name == "task_client" and module.startswith(
                            ("rclpy", "sensor_msgs", "cv2", "numpy")
                        ):
                            offenders.append((rel_path, module))

        self.assertEqual(offenders, [])

    def test_onboard_camera_node_does_not_import_task_or_motion_runtime(self):
        node_path = ROOT / "scripts" / "aufgabe04" / "qr_scanning" / "onboard_camera_node.py"
        source = node_path.read_text()
        tree = ast.parse(source, filename=str(node_path))
        forbidden_prefixes = (
            "scripts.aufgabe04.navigation",
            "scripts.aufgabe04.task_client",
            "scripts.aufgabe04.logistics",
            "geometry_msgs",
            "nav2_msgs",
        )
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [node.module or ""]
            else:
                continue
            for module in imported_modules:
                if module.startswith(forbidden_prefixes):
                    offenders.append(module)

        self.assertEqual(offenders, [])
        self.assertNotIn("/cmd_vel", source)
        self.assertNotIn("Twist", source)

    def test_stand_axis_analysis_stays_offline_and_motion_free(self):
        module_path = ROOT / "scripts" / "aufgabe04" / "perception" / "stand_axis_analysis.py"
        source = module_path.read_text()
        tree = ast.parse(source, filename=str(module_path))
        forbidden_prefixes = (
            "rclpy",
            "sensor_msgs",
            "geometry_msgs",
            "nav_msgs",
            "nav2_msgs",
            "rosbag2_py",
            "scripts.aufgabe04.navigation",
            "scripts.aufgabe04.logistics",
            "scripts.aufgabe04.fleet",
            "scripts.aufgabe04.stations",
            "scripts.aufgabe04.task_client",
            "scripts.aufgabe04.qr_scanning",
        )
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [node.module or ""]
            else:
                continue
            for module in imported_modules:
                if module.startswith(forbidden_prefixes):
                    offenders.append(module)

        self.assertEqual(offenders, [])
        self.assertNotIn("/cmd_vel", source)
        self.assertNotIn("Twist", source)
        self.assertNotIn("Publisher", source)

    def test_stand_axis_lidar_roi_stays_offline_and_motion_free(self):
        module_path = ROOT / "scripts" / "aufgabe04" / "perception" / "stand_axis_lidar_roi.py"
        source = module_path.read_text()
        tree = ast.parse(source, filename=str(module_path))
        forbidden_prefixes = (
            "rclpy",
            "sensor_msgs",
            "geometry_msgs",
            "nav_msgs",
            "nav2_msgs",
            "rosbag2_py",
            "scripts.aufgabe04.navigation",
            "scripts.aufgabe04.logistics",
            "scripts.aufgabe04.fleet",
            "scripts.aufgabe04.stations",
            "scripts.aufgabe04.task_client",
            "scripts.aufgabe04.qr_scanning",
        )
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [node.module or ""]
            else:
                continue
            for module in imported_modules:
                if module.startswith(forbidden_prefixes):
                    offenders.append(module)

        self.assertEqual(offenders, [])
        self.assertNotIn("/cmd_vel", source)
        self.assertNotIn("Twist", source)
        self.assertNotIn("Publisher", source)

    def test_navigation_runners_do_not_consume_stand_axis_lidar_roi_debug_artifact(self):
        checked_paths = [
            ROOT / "scripts" / "aufgabe04" / "navigation" / "plan_first_detected_station.py",
            ROOT / "scripts" / "aufgabe04" / "navigation" / "run_single_station_segment.py",
        ]
        forbidden_snippets = (
            "stand_axis_lidar_roi",
            "stand_axis_lidar_roi_observations.jsonl",
        )
        offenders = []
        for path in checked_paths:
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            for snippet in forbidden_snippets:
                if snippet in source:
                    offenders.append((path.relative_to(ROOT), snippet))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported_modules = [node.module or ""]
                else:
                    continue
                for module in imported_modules:
                    if module == "scripts.aufgabe04.perception.stand_axis_lidar_roi":
                        offenders.append((path.relative_to(ROOT), module))

        self.assertEqual(offenders, [])

    def test_stand_axis_viewer_stays_motion_free(self):
        module_path = ROOT / "scripts" / "aufgabe04" / "perception" / "debug" / "stand_axis_viewer.py"
        source = module_path.read_text()
        tree = ast.parse(source, filename=str(module_path))
        forbidden_prefixes = (
            "geometry_msgs",
            "nav2_msgs",
            "scripts.aufgabe04.navigation",
            "scripts.aufgabe04.logistics",
            "scripts.aufgabe04.fleet",
            "scripts.aufgabe04.stations",
            "scripts.aufgabe04.task_client",
        )
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [node.module or ""]
            else:
                continue
            for module in imported_modules:
                if module.startswith(forbidden_prefixes):
                    offenders.append(module)

        self.assertEqual(offenders, [])
        self.assertNotIn("--cmd-vel-topic", source)
        self.assertNotIn("/cmd_vel", source)
        self.assertNotIn("create_publisher", source)
        self.assertNotIn("Twist", source)
        self.assertNotIn("Publisher", source)

    def test_structural_detection_and_tracking_modules_stay_pure(self):
        checked_paths = [
            ROOT
            / "scripts"
            / "aufgabe04"
            / "perception"
            / "stand_structure_hypothesis.py",
            ROOT
            / "scripts"
            / "aufgabe04"
            / "perception"
            / "stand_axis_tracking.py",
        ]
        forbidden_prefixes = (
            "rclpy",
            "sensor_msgs",
            "geometry_msgs",
            "nav2_msgs",
            "scripts.aufgabe04.perception.debug",
            "scripts.aufgabe04.navigation",
            "scripts.aufgabe04.logistics",
            "scripts.aufgabe04.fleet",
            "scripts.aufgabe03",
        )
        offenders = []
        for path in checked_paths:
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported_modules = [node.module or ""]
                else:
                    continue
                for module in imported_modules:
                    if module.startswith(forbidden_prefixes):
                        offenders.append((path.name, module))
            self.assertNotIn("create_publisher", source)
            self.assertNotIn("/cmd_vel", source)

        self.assertEqual(offenders, [])

    def test_stand_axis_internal_package_stays_pure_and_motion_free(self):
        package_root = (
            ROOT / "scripts" / "aufgabe04" / "perception" / "stand_axis"
        )
        if not package_root.exists():
            self.fail("stand-axis implementation package is missing")

        forbidden_prefixes = (
            "rclpy",
            "sensor_msgs",
            "geometry_msgs",
            "nav2_msgs",
            "scripts.aufgabe04.perception.debug",
            "scripts.aufgabe04.perception.stand_axis_consensus",
            "scripts.aufgabe04.perception.stand_axis_tracking",
            "scripts.aufgabe04.navigation",
            "scripts.aufgabe04.logistics",
            "scripts.aufgabe04.fleet",
            "scripts.aufgabe04.stations",
            "scripts.aufgabe04.real_robot",
            "scripts.aufgabe03",
        )
        forbidden_source = (
            "create_publisher",
            "/cmd_vel",
            "run_simple_waypoint_follower",
            "run_single_station_segment",
        )
        offenders = []
        for path in sorted(package_root.glob("*.py")):
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported_modules = [node.module or ""]
                else:
                    continue
                for module in imported_modules:
                    if module.startswith(forbidden_prefixes):
                        offenders.append((path.name, module))
            for snippet in forbidden_source:
                if snippet in source:
                    offenders.append((path.name, snippet))

        self.assertEqual(offenders, [])

    def test_passive_real_viewpoint_observer_stays_motion_free(self):
        module_path = (
            ROOT
            / "scripts"
            / "aufgabe04"
            / "real_robot"
            / "passive_viewpoint_node.py"
        )
        source = module_path.read_text()
        tree = ast.parse(source, filename=str(module_path))
        forbidden_prefixes = (
            "geometry_msgs",
            "nav2_msgs",
            "scripts.aufgabe04.navigation.run_single_station_segment",
            "scripts.aufgabe04.navigation.simple_waypoint_follower",
        )
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [node.module or ""]
            else:
                continue
            for module in imported_modules:
                if module.startswith(forbidden_prefixes):
                    offenders.append(module)

        self.assertEqual(offenders, [])
        for snippet in (
            "create_publisher",
            "--cmd-vel-topic",
            "/cmd_vel",
            "stand_axis_tracking",
            "Twist",
            "Publisher",
        ):
            self.assertNotIn(snippet, source)

    def test_navigation_runners_do_not_consume_structural_diagnostics(self):
        checked_paths = [
            ROOT / "scripts" / "aufgabe04" / "navigation" / "plan_first_detected_station.py",
            ROOT / "scripts" / "aufgabe04" / "navigation" / "run_single_station_segment.py",
        ]
        offenders = []
        for path in checked_paths:
            source = path.read_text()
            for snippet in (
                "stand_structure_hypothesis",
                "stand_structure ",
                "structural-diagnostic",
            ):
                if snippet in source:
                    offenders.append((path.relative_to(ROOT), snippet))

        self.assertEqual(offenders, [])

    def test_autonomous_parent_does_not_import_private_coverage_helpers(self):
        parent_path = (
            ROOT
            / "scripts"
            / "aufgabe04"
            / "real_robot"
            / "run_autonomous_stand_exploration.py"
        )
        tree = ast.parse(parent_path.read_text(), filename=str(parent_path))
        extracted_modules = {
            "scripts.aufgabe04.real_robot.autonomous_coverage_execution",
            "scripts.aufgabe04.real_robot.autonomous_coverage_replanning",
            "scripts.aufgabe04.real_robot.autonomous_candidate_approach",
        }
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if (node.module or "") not in extracted_modules:
                continue
            offenders.extend(
                (node.module, alias.name)
                for alias in node.names
                if alias.name.startswith("_")
            )

        self.assertEqual(offenders, [])

    def test_autonomous_parent_only_imports_declared_phase_apis(self):
        real_robot_root = ROOT / "scripts" / "aufgabe04" / "real_robot"
        module_paths = {
            "scripts.aufgabe04.real_robot.autonomous_coverage_execution": (
                real_robot_root / "autonomous_coverage_execution.py"
            ),
            "scripts.aufgabe04.real_robot.autonomous_coverage_replanning": (
                real_robot_root / "autonomous_coverage_replanning.py"
            ),
            "scripts.aufgabe04.real_robot.autonomous_candidate_approach": (
                real_robot_root / "autonomous_candidate_approach.py"
            ),
        }
        declared_exports = {}
        for module_name, module_path in module_paths.items():
            tree = ast.parse(module_path.read_text(), filename=str(module_path))
            export_assignment = next(
                (
                    node
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "__all__"
                        for target in node.targets
                    )
                ),
                None,
            )
            self.assertIsNotNone(export_assignment, module_name)
            declared_exports[module_name] = set(
                ast.literal_eval(export_assignment.value)
            )

        parent_path = real_robot_root / "run_autonomous_stand_exploration.py"
        parent_tree = ast.parse(
            parent_path.read_text(),
            filename=str(parent_path),
        )
        offenders = []
        for node in ast.walk(parent_tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module_name = node.module or ""
            if module_name not in declared_exports:
                continue
            offenders.extend(
                (module_name, alias.name)
                for alias in node.names
                if alias.name not in declared_exports[module_name]
            )

        self.assertEqual(offenders, [])

    def test_extracted_autonomous_phases_do_not_own_live_edges(self):
        module_names = (
            "autonomous_coverage_replanning.py",
            "autonomous_candidate_approach.py",
        )
        forbidden_imports = (
            "rclpy",
            "sensor_msgs",
            "geometry_msgs",
            "nav_msgs",
            "nav2_msgs",
            "subprocess",
            "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration",
        )
        forbidden_source = (
            "create_publisher",
            "/cmd_vel",
            "input(",
        )
        offenders = []
        for module_name in module_names:
            module_path = (
                ROOT / "scripts" / "aufgabe04" / "real_robot" / module_name
            )
            self.assertTrue(module_path.is_file(), module_path)
            source = module_path.read_text()
            tree = ast.parse(source, filename=str(module_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported_modules = [node.module or ""]
                else:
                    continue
                for imported_module in imported_modules:
                    if imported_module.startswith(forbidden_imports):
                        offenders.append((module_name, imported_module))
            for snippet in forbidden_source:
                if snippet in source:
                    offenders.append((module_name, snippet))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
