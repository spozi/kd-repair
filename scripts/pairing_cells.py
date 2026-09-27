"""Locate the inputs of one teacher-student pairing at one dataset-profile setting.

A pairing names a teacher architecture and a student architecture. Its cells are the 52
Experiment 1 settings. For every cell this module finds the study spec, the original teacher,
the repaired teacher when one was selected, the target classes, and each seed's
classical-KD control, whichever script produced them:

- cnn-cnn          Experiment 1 itself (cifar_teacher -> cifar_student)
- cnn-resnet8/20   student_architecture_study.py (original-teacher students are the controls)
- r18-cnn, r34-cnn teacher_architecture_study.py (its baselines hold teacher and controls)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kd.checkpoints import fingerprint

PAIRINGS = {
    "cnn-cnn": ("cifar_teacher", "cifar_student"),
    "cnn-resnet8": ("cifar_teacher", "resnet8"),
    "cnn-resnet20": ("cifar_teacher", "resnet20"),
    "r18-cnn": ("cifar_resnet18", "cifar_student"),
    "r34-cnn": ("cifar_resnet34", "cifar_student"),
}
TEACHER_SUFFIX = {"cifar_resnet18": "r18", "cifar_resnet34": "r34"}


def _read(path: Path):
    return json.loads(path.read_text())


@dataclass
class Cell:
    pairing: str
    job: str
    root: Path            # directory holding baselines/confirmatory and studies/confirmatory
    spec: dict            # teacher_run, student_run_template, student_seeds, evaluation_split
    status: str           # "complete" or "no_repair_selected"
    target_classes: list
    teacher: Path         # original teacher checkpoint
    repaired: Path | None
    student_root: Path | None = None   # where cnn-resnet controls live

    def control(self, seed: int) -> Path:
        """Directory of the classical-KD student distilled from the original teacher."""
        student = PAIRINGS[self.pairing][1]
        if self.student_root is not None:
            matches = sorted(self.student_root.glob(f"*/{self.job}/{student}_original_{self.job}_seed{seed}"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one {student} control for {self.job} seed {seed}, found {matches}")
            return matches[0]
        return self.root / "baselines" / "confirmatory" / self.spec["student_run_template"].format(seed=seed)

    def reference_data(self, seed: int) -> dict | None:
        """Test-split identity recorded when the study scored its students, if it did."""
        path = (self.root / "studies" / "confirmatory" / "student_evaluation" / f"seed{seed}"
                / "baseline" / "predictions.json")
        return _read(path)["identity"]["data"] if path.is_file() else None


def cells(pairing: str, matrix: Path, *, teachers: Path | None = None, students: Path | None = None,
          jobs: list[str] | None = None, verify_repaired: bool = True) -> list[Cell]:
    """Every finished cell of a pairing, in matrix order, optionally restricted to some jobs.

    verify_repaired=False skips checking the selected repaired teacher, for uses that only
    need the original teacher and its controls.
    """
    teacher_model, student_model = PAIRINGS[pairing]
    plan = matrix / "matrix-plan" / "configs"
    out = []
    for job in _read(matrix / "matrix_summary.json")["jobs"]:
        if jobs and job["id"] not in jobs:
            continue
        if teacher_model == "cifar_teacher":
            root, spec_path = matrix / job["dataset"] / job["profile"], plan / f"{job['id']}.json"
        else:
            if teachers is None:
                raise ValueError("ResNet-teacher pairings need the teacher-architecture output root")
            name = f"{job['id']}-{TEACHER_SUFFIX[teacher_model]}"
            root, spec_path = teachers / name, teachers / "configs" / f"{name}.json"
        comparison_path = root / "studies" / "confirmatory" / "comparison.json"
        if not comparison_path.is_file():
            continue  # this pairing's repair decision has not been made yet
        spec = _read(spec_path)
        comparison = _read(comparison_path)
        if comparison.get("status") not in {"complete", "no_repair_selected"}:
            continue
        teacher = root / "baselines" / "confirmatory" / spec["teacher_run"] / "student.pt"
        repaired = None
        if comparison["status"] == "complete":
            selected = comparison["teacher_selection"]["selected"]
            repaired = (root / "studies" / "confirmatory" / "repair_candidates"
                        / f"budget{selected['budget']}_lr{selected['learning_rate']}" / "teacher.pt")
            if verify_repaired and (not repaired.is_file()
                                    or fingerprint(repaired) != selected["checkpoint_sha256"]):
                raise ValueError(f"Selected repaired teacher missing or changed: {repaired}")
        student_root = None
        if student_model != "cifar_student" and teacher_model == "cifar_teacher":
            if students is None:
                raise ValueError("cnn-resnet pairings need the student-architecture output root")
            student_root = students
        out.append(Cell(pairing, job["id"], root, spec, comparison["status"],
                        comparison["target_classes"], teacher, repaired, student_root))
    return out
