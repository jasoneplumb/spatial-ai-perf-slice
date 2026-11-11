"""Release checks that need no GPU, OpenPCDet, dataset or network access.

Run from the repository root:

    python -m unittest discover -s tests -v

They cover Python and shell syntax, determinism of the synthetic scan
generator, and repository invariants (nothing restricted is tracked, pins and
versions agree, documentation references resolve). They do not exercise the
CUDA profiling tools.
"""
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

RELEASE_VERSION = "0.2.0"
# Recorded in docs/BOUNDARY-INVENTORY.md (seed-7 scan, default parameters; waymo-like preset total).
SEED7_SHA256 = "c9d636f325787e6c2a90b6b70bb10bfe5e1a64f8c161544a8fd3bae9c4c75459"
SEED7_POINTS = 158436
SEED7_WAYMO_LIKE_POINTS = 182946

# Datasets, checkpoints, exported models and other generated binaries stay out of version control.
FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".onnx", ".npy", ".npz", ".pkl", ".pickle", ".bin", ".tfrecord", ".h5", ".hdf5",
    ".safetensors", ".engine", ".trt", ".pb", ".zip", ".tar", ".gz", ".7z", ".so", ".whl",
}
IGNORED_DIRS = ("third_party/", "artifacts/", "results/")
MAX_TRACKED_BYTES = 256 * 1024
SYNTHETIC_RESULT_TOOLS = {
    "cpu_sparse_backbone.py", "density_sweep.py", "export_dense_onnx.py", "inventory_boundaries.py",
    "nms_sweep.py", "pillar_variant.py", "smoke_centerpoint.py",
}


def tracked_files():
    out = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"], text=True)
    # A file deleted in the working tree but not yet committed is still listed.
    return [p for p in out.split("\0") if p and (ROOT / p).is_file()]


TRACKED = tracked_files()
MARKDOWN = [p for p in TRACKED if p.endswith(".md")]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


class Syntax(unittest.TestCase):
    def test_python_sources_parse(self):
        sources = [p for p in TRACKED if p.endswith(".py")]
        self.assertTrue(sources)
        for path in sources:
            with self.subTest(path=path):
                ast.parse(read(path), filename=path)

    @unittest.skipUnless(shutil.which("bash"), "bash not available")
    def test_shell_scripts_parse(self):
        scripts = [p for p in TRACKED if p.endswith(".sh")]
        self.assertTrue(scripts)
        for path in scripts:
            with self.subTest(path=path):
                subprocess.run(["bash", "-n", str(ROOT / path)], check=True)


class ScanGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import scan_generator

        cls.sg = scan_generator
        cls.points, cls.scene = scan_generator.generate(scan_generator.ScanParams(), 7)

    def test_output_is_model_ready(self):
        import numpy as np

        self.assertEqual(self.points.ndim, 2)
        self.assertEqual(self.points.shape[1], 5)
        self.assertEqual(self.points.dtype, np.float32)
        self.assertTrue(np.isfinite(self.points).all())
        self.assertEqual(self.scene["returns"], self.points.shape[0])

    def test_same_seed_is_bit_identical(self):
        again, scene = self.sg.generate(self.sg.ScanParams(), 7)
        self.assertEqual(again.tobytes(), self.points.tobytes())
        self.assertEqual(scene, self.scene)

    def test_different_seed_differs(self):
        other, _ = self.sg.generate(self.sg.ScanParams(), 8)
        self.assertNotEqual(other.tobytes(), self.points.tobytes())

    def test_default_scan_matches_documented_hash(self):
        self.assertEqual(self.points.shape[0], SEED7_POINTS)
        self.assertEqual(hashlib.sha256(self.points.tobytes()).hexdigest(), SEED7_SHA256)
        self.assertIn(SEED7_SHA256[:8], read("docs/BOUNDARY-INVENTORY.md"))

    def test_waymo_like_preset_matches_documented_count(self):
        params = self.sg.ScanParams(**self.sg.PRESETS["waymo-like"])
        points, scene = self.sg.generate(params, 7)
        self.assertEqual(points.shape[0], SEED7_WAYMO_LIKE_POINTS)
        self.assertEqual(set(scene["per_sensor"]), {"top", "front", "side_left", "side_right", "rear"})
        self.assertIn(f"{SEED7_WAYMO_LIKE_POINTS:,}", read("docs/BOUNDARY-INVENTORY.md"))

    def test_cli_writes_scan_and_matching_sidecar(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scan.npy"
            subprocess.run([sys.executable, str(ROOT / "tools" / "scan_generator.py"), "--seed", "7", "--out", str(out)],
                           check=True, stdout=subprocess.DEVNULL)
            meta = json.loads(out.with_suffix(".json").read_text())
            saved = np.load(out)
        self.assertEqual(meta["points_sha256"], SEED7_SHA256)
        self.assertEqual(hashlib.sha256(saved.tobytes()).hexdigest(), SEED7_SHA256)
        self.assertEqual(meta["generator_version"], self.sg.GENERATOR_VERSION)


class RepositoryInvariants(unittest.TestCase):
    def test_no_restricted_or_generated_content_is_tracked(self):
        for path in TRACKED:
            with self.subTest(path=path):
                self.assertNotIn(Path(path).suffix.lower(), FORBIDDEN_SUFFIXES)
                self.assertFalse(path.startswith(("third_party/", "artifacts/")))
                self.assertTrue(not path.startswith("results/") or path == "results/README.md")
                self.assertLessEqual((ROOT / path).stat().st_size, MAX_TRACKED_BYTES)

    def test_gitignore_excludes_external_and_generated_directories(self):
        ignored = read(".gitignore").split()
        for entry in IGNORED_DIRS:
            self.assertIn(entry, ignored)

    def test_license_files(self):
        text = read("LICENSE")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0, January 2004", text)
        self.assertIn("Apache License, Version 2.0", read("NOTICE"))
        self.assertIn("Waymo Dataset License Agreement", read("THIRD_PARTY_NOTICES.md"))

    def test_openpcdet_pin_is_consistent(self):
        pin = re.search(r'OPENPCDET_COMMIT="([0-9a-f]{40})"', read("scripts/setup-openpcdet.sh")).group(1)
        mentions = 0
        for path in TRACKED:
            if not path.endswith((".md", ".sh", ".py")):
                continue
            for line in read(path).splitlines():
                if "openpcdet" in line.lower():
                    for sha in re.findall(r"\b[0-9a-f]{40}\b", line):
                        mentions += 1
                        self.assertEqual(sha, pin, f"{path}: {line.strip()}")
        self.assertGreaterEqual(mentions, 3)
        self.assertIn(pin, read("README.md"))

    def test_release_version_is_consistent(self):
        first = re.search(r"^## \[([^\]]+)\]", read("CHANGELOG.md"), re.MULTILINE).group(1)
        self.assertEqual(first, RELEASE_VERSION)
        self.assertIn(f"`v{RELEASE_VERSION}`", read("README.md"))
        for path in MARKDOWN:
            with self.subTest(path=path):
                self.assertIsNone(re.search(re.escape(RELEASE_VERSION) + r"-[A-Za-z]", read(path)))

    def test_synthetic_results_are_never_marked_as_baselines(self):
        marked = set()
        for path in TRACKED:
            if path.startswith("tools/") and path.endswith(".py"):
                flags = re.findall(r'"valid_for_baseline":\s*(\w+)', read(path))
                if flags:
                    marked.add(Path(path).name)
                    self.assertEqual(set(flags), {"False"}, path)
        self.assertLessEqual(SYNTHETIC_RESULT_TOOLS, marked)

    def test_markdown_links_and_repo_paths_resolve(self):
        link = re.compile(r"\]\(([^)\s]+)\)")
        repo_path = re.compile(r"(?<![\w/.-])((?:tools|scripts|docs|tests)/[\w.-]+\.(?:py|sh|ps1|md))\b")
        for path in MARKDOWN:
            text = read(path)
            for target in link.findall(text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                with self.subTest(path=path, link=target):
                    self.assertTrue((ROOT / path).parent.joinpath(target.split("#")[0]).exists())
            for target in repo_path.findall(text):
                with self.subTest(path=path, repo_path=target):
                    self.assertTrue((ROOT / target).is_file())


if __name__ == "__main__":
    unittest.main()
