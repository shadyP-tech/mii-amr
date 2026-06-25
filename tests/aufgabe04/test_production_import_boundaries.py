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


if __name__ == "__main__":
    unittest.main()
