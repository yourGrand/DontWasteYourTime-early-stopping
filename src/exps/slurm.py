from __future__ import annotations

import os
import re
import sys
import shutil
import argparse
import inspect
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar, overload
from typing_extensions import Self, override

import pandas as pd

from exps.parsable import Arg, Parsable

try:
    from rich_argparse import MetavarTypeRichHelpFormatter

    HelpFormatter = MetavarTypeRichHelpFormatter
except ImportError:
    HelpFormatter = argparse.MetavarTypeHelpFormatter


def _try_interpret_python_path() -> Path:
    import os

    # Prefer virtual environment 
    if (path := os.environ.get("VIRTUAL_ENV")) is not None:
        path = Path(path).resolve().expanduser()
        path = path / "bin" / "python"
    elif (path := os.environ.get("PYTHON_PATH")) is not None or (
        path := os.environ.get("CONDA_PYTHON_EXE")
    ) is not None:
        path = Path(path).resolve().expanduser()
    else:
        raise ValueError(
            "Could not find a Python path, please provide explicitly",
        )

    if not path.exists():
        raise ValueError(
            f"Trying to get python path {path} but it does not exist"
            " Please provide an explicit python path",
        )
    return path


def _try_interpret_python_path_v2() -> Path:

    # First priority: use the current Python interpreter (most reliable)
    path = Path(sys.executable).resolve()
    if path.exists():
        return path

    # Second priority: check environment variables
    if (path := os.environ.get("PYTHON_PATH")) is not None:
        path = Path(path).resolve().expanduser()
        if path.exists():
            return path
    
    # Third priority: check conda-specific environment variables
    if (path := os.environ.get("CONDA_PYTHON_EXE")) is not None:
        path = Path(path).resolve().expanduser()
        if path.exists():
            return path
    
    if (conda_prefix := os.environ.get("CONDA_PREFIX")) is not None:
        path = Path(conda_prefix) / "bin" / "python"
        if path.exists():
            return path
    
    # Fourth priority: check virtual env variables
    if (path := os.environ.get("VIRTUAL_ENV")) is not None:
        path = Path(path).resolve().expanduser() / "bin" / "python"
        if path.exists():
            return path
    
    # Use which command
    try:
        path_str = subprocess.check_output(["which", "python"], text=True).strip()
        path = Path(path_str).resolve()
        if path.exists():
            return path
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    raise ValueError(
        "Could not find a Python path, please provide explicitly",
    )


def as_slurm_header(headers: dict[str, Any]) -> str:
    return "\n".join([f"#SBATCH --{k}={v}" for k, v in headers.items()])


def seconds_to_slurm_time(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass(kw_only=True)
class Slurmable(Parsable):
    GROUPS_FOR_PATH: ClassVar[tuple[str, ...]] = ()

    # -- paths
    root: Path = field(
        metadata=Arg(help="Root directory", group="paths"),
        default=Path("./results").resolve(),
    )
    unique_path: Path = field(init=False)

    def __post_init__(self) -> None:
        if len(self.GROUPS_FOR_PATH) == 0:
            raise ValueError(
                "Please specify the GROUPS_FOR_PATH class variable with at least one"
                " group",
            )

        grouped_fields = self.grouped_fields(order=self.GROUPS_FOR_PATH)

        p = Path(self.root)
        for _, group_fields in grouped_fields.items():
            p = p / "-".join(f"{k}={v}" for k, v, _ in group_fields)

        self.unique_path = p.resolve().expanduser()

    @classmethod
    def as_array(cls, items: list[Self]) -> ArraySlurmable[Self]:
        return ArraySlurmable(items)

    def flag(
        self,
        status: Literal["running", "failed", "success", "submitted", "pending"],
    ) -> Path:
        return self.unique_path / f".flag.{status}"

    @classmethod
    def _run_file(cls) -> Path:
        return Path(inspect.getfile(cls))

    def exec_script(
        self,
        python: Path | str | None = None,
        slurm_headers: dict[str, Any] | None = None,
    ) -> str:
        headers = as_slurm_header(slurm_headers) if slurm_headers else ""

        field_values = self.item_fields()
        if python is None:
            python = _try_interpret_python_path_v2()

        running_flag = self.flag("running").resolve().absolute()
        failed_flag = self.flag("failed").resolve().absolute()
        success_flag = self.flag("success").resolve().absolute()
        exec_file = self._run_file().resolve().absolute()
        exec_str = " \\\n  ".join(
            [
                f"{python} {exec_file}",
                *[f'--{k} "{v}"' for k, (v, _) in field_values.items()],
                f"&& touch {success_flag} || touch {failed_flag}",
            ],
        )
        return "\n".join(
            [
                "#!/bin/bash",
                headers,
                "\n",
                "set -e",
                "set -u",
                "set -o pipefail",
                "\n" f"touch {running_flag}",
                exec_str,
                "\n",
            ],
        )

    def run(
        self,
        *,
        python: Path | str | None = None,
        bash: Path | str = "/usr/bin/bash",
    ) -> None:
        now = datetime.now().isoformat()
        exec_script = self.exec_script(python=python, slurm_headers={})
        exec_script_path = self.unique_path / f"run-{now}.sh"
        exec_script_path.parent.mkdir(parents=True, exist_ok=True)
        with exec_script_path.open("w") as f:
            f.write(exec_script)

        print(exec_script)

        bash_path = shutil.which("bash")
        if bash_path is None:
            raise ValueError("Cannot find 'bash' executable in PATH")
        bash = Path(bash_path)

        subprocess.run([str(bash), str(exec_script_path)], check=True)  # noqa: S603

    def submit(
        self,
        slurm_headers: dict[str, Any],
        *,
        python: Path | str | None = None,
        sbatch: Path | str = "sbatch",
    ) -> None:
        if "job-name" not in slurm_headers:
            jobname = f"{self.__class__.__name__}"
            slurm_headers["job-name"] = jobname

        now = datetime.now().isoformat()
        exec_script = self.exec_script(python=python, slurm_headers=slurm_headers)
        exec_script_path = self.unique_path / f"run-{now}.sh"
        exec_script_path.parent.mkdir(parents=True, exist_ok=True)
        with exec_script_path.open("w") as f:
            f.write(exec_script)

        self.flag("submitted").touch()
        subprocess.run([str(sbatch), str(exec_script_path)], check=True)  # noqa: S603

    def status(self) -> Literal["failed", "running", "pending", "success", "submitted"]:
        if self.flag("success").exists():
            return "success"
        if self.flag("failed").exists():
            return "failed"
        if self.flag("running").exists():
            return "running"
        if self.flag("submitted").exists():
            return "submitted"
        return "pending"

    @classmethod
    def add_submission_arguments(
        cls,
        parser: argparse.ArgumentParser,
    ) -> argparse._ArgumentGroup:
        group = parser.add_argument_group("Slurm submission")
        group.add_argument(
            "--job-name",
            type=str,
            required=True,
            help="Name of the job",
        )
        group.add_argument(
            "--partition",
            type=str,
            required=True,
            help="Partition to submit to",
        )
        group.add_argument(
            "--gres",
            type=str,
            required=False,
            help="Any additional gres",
        )

        group.add_argument(
            "--mem",
            type=str,
            required=True,
            help="Memory total",
        )
        group.add_argument(
            "--time",
            type=str,
            required=True,
            help="Time for the job",
        )
        group.add_argument(
            "--cpus-per-task",
            type=int,
            default=1,
            help="Number of cpus per task",
        )

        group.add_argument(
            "--script-dir",
            type=Path,
            default=None,
            help=(
                "Directory to write the submission script to."
                " Defaults to `./slurm-scripts`"
            ),
        )
        group.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit the number of jobs to submit",
        )
        group.add_argument(
            "--python",
            type=Path,
            default=None,
            help=(
                "Path to the python executable, otherwise will try to interptet it"
                "from the current activated env."
            ),
        )
        group.add_argument(
            "--log-dir",
            type=Path,
            default=Path("./slurm-log-dir"),
            help="Log directory, defaults `./slurm-log-dir`",
        )
        group.add_argument(
            "--output",
            type=str,
            default="%j-%a.out",
            help=(
                "The output file for stdout. Defaults to the"
                " `<jobid>-<arrayid>.out` places in the --log-dir"
            ),
        )
        group.add_argument(
            "--error",
            type=str,
            default="%j-%a.err",
            help=(
                "The error file for stderr. Defaults to the"
                " `<jobid>-<arrayid>.err` places in the --log-dir"
            ),
        )

        group.add_argument(
            "--sbatch-cmd",
            type=str,
            default="sbatch",
            help="Overwrite the sbatch command if required.",
        )
        group.add_argument(
            "--mail-type",
            type=str,
            default=None,
            help="Mail type",
        )
        group.add_argument(
            "--mail-user",
            type=str,
            default=None,
            help="Mail User",
        )
        return group

    def reset(self) -> None:
        self.flag("running").unlink(missing_ok=True)
        self.flag("failed").unlink(missing_ok=True)
        self.flag("success").unlink(missing_ok=True)
        self.flag("submitted").unlink(missing_ok=True)


T = TypeVar("T", bound=Slurmable)


@dataclass
class ArraySlurmable(Sequence[T]):
    items: Sequence[T]

    def __post_init__(self) -> None:
        if len(self.items) == 0:
            raise ValueError("Array must have at least one item")

        first = self.items[0]
        first_root = first.root
        if any(item.root != first_root for item in self.items):
            raise ValueError("All items must have the same root")

        if any(item.__class__ != first.__class__ for item in self.items):
            raise ValueError("All items must have the same class")

    @overload
    def __getitem__(self, i: int) -> T:
        ...

    @overload
    def __getitem__(self, i: slice) -> Sequence[T]:
        ...

    @override
    def __getitem__(self, i: int | slice) -> T | Sequence[T]:
        return self.items[i]

    @override
    def __len__(self) -> int:
        return len(self.items)

    def submission_script(
        self,
        name: str,
        slurm_headers: dict[str, Any],
        *,
        python: Path | str | None = None,
        generate_item_scripts: bool = False,
        limit: int | None = None,
        start_idx: int | None = None,
        end_idx: int | None = None,
    ) -> str:
        headers = dict(slurm_headers)

        assert "job-name" not in slurm_headers
        jobname = f"{name}-array"
        headers["job-name"] = jobname

        headers_without_array = {k: v for k, v in headers.items() if k != "array"}
        now = datetime.now().isoformat()

        paths = []
        # Write the exec scripts to each item
        for item in self.items:
            exc_script = item.exec_script(
                python=python,
                slurm_headers=headers_without_array,
            )

            exec_script_path = item.unique_path / f"{name}-array-submit-{now}.sh"

            if generate_item_scripts:
                exec_script_path.parent.mkdir(parents=True, exist_ok=True)
                with exec_script_path.open("w") as f:
                    f.write(exc_script)

            paths.append(exec_script_path)

        assert "array" not in slurm_headers
        
        # headers["array"] = f"0-{len(self.items) - 1}"
        
        # if limit is not None:
        #     # assert "array" not in slurm_headers
        #     lim = min(limit, len(self.items))
        #     headers["array"] = f"0-{len(self.items) - 1}%{lim}"
            
        # if start_idx is not None and end_idx is not None:
        #     chunk_size = end_idx - start_idx + 1
            
        #     if limit is not None:
        #         chunk_limit = min(limit, chunk_size)
        #         chunk_limit_spec = f"%{chunk_limit}"
                
        #         headers["array"] = f"0-{chunk_size-1}{chunk_limit_spec}"
        #     else:
        #         headers["array"] = f"0-{chunk_size-1}"
            
        #     paths = paths[start_idx:end_idx + 1]
            
        #     # else:
        #         # headers["array"] = f"0-{len(self.items) - 1}%{lim}"
        
        array_config = {}
        
        # Range
        if start_idx is not None and end_idx is not None:
            chunk_size = end_idx - start_idx + 1
            array_config["range"] = f"0-{chunk_size-1}"
            paths = paths[start_idx:end_idx + 1]
        else:
            array_config["range"] = f"0-{len(self.items) - 1}"
        
        # Limit
        if limit is not None:
            if start_idx is not None and end_idx is not None:
                max_concurrent = min(limit, end_idx - start_idx + 1)
            else:
                max_concurrent = min(limit, len(self.items))
                
            array_config["limit"] = max_concurrent
        
        if "limit" in array_config:
            headers["array"] = f"{array_config['range']}%{array_config['limit']}"
        else:
            headers["array"] = array_config["range"]
        
        header_str = as_slurm_header(headers) 

        _items_str = "\n  ".join([f'"{p}"' for p in paths])
        items_arr = f"script_paths=(\n  {_items_str}\n)"
        selected_task = 'selected_task="${script_paths[$SLURM_ARRAY_TASK_ID]}"'
        echo_task_running = 'echo "Running ${selected_task}"'
        echo_break = 'echo "-----------------"'
        cat_exc = 'cat "${selected_task}"'
        echo_break = 'echo "-----------------"'
        bash_exc = 'bash "${selected_task}"'

        return "\n".join(
            [
                "#!/bin/bash",
                header_str,
                "\n",
                "set -e",
                "set -u",
                "set -o pipefail",
                "\n",
                f"echo '{name}'",
                items_arr,
                "\n",
                selected_task,
                echo_task_running,
                echo_break,
                cat_exc,
                echo_break,
                bash_exc,
            ],
        )

    def submit(
        self,
        name: str,
        slurm_headers: dict[str, Any],
        *,
        sbatch: str | list[str] = "sbatch",
        script_dir: Path | None = None,
        python: Path | str | None = None,
        limit: int | None = None,
        job_array_limit: int | None = 1080,
    ) -> None:
        
        total_items = len(self.items)
        now = datetime.now().isoformat()
        
        script_dir = script_dir or Path()
        script_dir = script_dir.absolute().resolve()
        script_dir.mkdir(parents=True, exist_ok=True)
        
        _sbatch = [sbatch] if isinstance(sbatch, str) else sbatch
        
        # Complex case: split into chunks with dependencies
        if job_array_limit is not None and total_items > job_array_limit:
            for start_idx in range(0, total_items, job_array_limit):
                end_idx = min(start_idx + job_array_limit - 1, total_items - 1)
                chunk_name = f"{name}-chunk-{start_idx}-{end_idx}"
                
                submission_script = self.submission_script(
                    chunk_name,
                    slurm_headers,
                    python=python,
                    generate_item_scripts=True,
                    limit=limit,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
                
                script_path = script_dir / f"{chunk_name}-array-submit-{now}.sh"
                with script_path.open("w") as f:
                    f.write(submission_script)
                
                print(f"Submitting chunk {start_idx}-{end_idx} of {total_items-1}")
                
                try:
                    subprocess.run([*_sbatch, str(script_path)], check=True)  # noqa: S603
                except Exception as e:
                    # print(
                    #     "Debugging off SLURM cluster. Error:" +
                    #     "\n-----------------------------\n" +
                    #     f"{e}" +
                    #     "\n-----------------------------\n"
                    #     )
                    raise e

        # Simple case: submit as a single job array
        else:
            submission_script = self.submission_script(
                name,
                slurm_headers,
                python=python,
                generate_item_scripts=True,
                limit=limit,
            )
            
            script_path = script_dir / f"{name}-array-submit-{now}.sh"
            with script_path.open("w") as f:
                f.write(submission_script)
                
            print(f"Submitting all {total_items} jobs as a single array")
            subprocess.run([*_sbatch, str(script_path)], check=True)  # noqa: S603
    
    # def submit(
    #     self,
    #     name: str,
    #     slurm_headers: dict[str, Any],
    #     *,
    #     sbatch: str | list[str] = "sbatch",
    #     script_dir: Path | None = None,
    #     python: Path | str | None = None,
    #     limit: int | None = None,
    #     job_array_limit: int | None = 1080,
    # ) -> None:
        
    #     total_items = len(self.items)
    #     now = datetime.now().isoformat()
        
    #     script_dir = script_dir or Path()
    #     script_dir = script_dir.absolute().resolve()
    #     script_dir.mkdir(parents=True, exist_ok=True)
        
    #     _sbatch = [sbatch] if isinstance(sbatch, str) else sbatch
    #     if job_array_limit == None or total_items <= job_array_limit:
    #         now = datetime.now().isoformat()
    #         submission_script = self.submission_script(
    #             name,
    #             slurm_headers,
    #             python=python,
    #             generate_item_scripts=True,
    #             limit=limit,
    #         )
    #         script_dir = script_dir or Path()
    #         script_dir = script_dir.absolute().resolve()
    #         script_dir.mkdir(parents=True, exist_ok=True)

    #         script_path = script_dir / f"{name}-array-submit-{now}.sh"
    #         with script_path.open("w") as f:
    #             f.write(submission_script)

    #         _sbatch = [sbatch] if isinstance(sbatch, str) else sbatch
    #         subprocess.run([*_sbatch, str(script_path)], check=True)  # noqa: S603
    #     else:
    #         # Split into chunks
    #         for start_idx in range(0, total_items, job_array_limit):
    #             end_idx = min(start_idx + job_array_limit - 1, total_items - 1)
    #             chunk_name = f"{name}-chunk-{start_idx}-{end_idx}"
                
    #             submission_script = self.submission_script(
    #                 chunk_name,
    #                 slurm_headers,
    #                 python=python,
    #                 generate_item_scripts=True,
    #                 limit=limit,
    #                 start_idx=start_idx,
    #                 end_idx=end_idx,
    #             )
                
    #             script_path = script_dir / f"{chunk_name}-array-submit-{now}.sh"
    #             with script_path.open("w") as f:
    #                 f.write(submission_script)
                
    #             print(f"Submitting chunk {start_idx}-{end_idx} of {total_items-1}")
                
    #             # TEMP
    #             try:
    #                 subprocess.run([*_sbatch, str(script_path)], check=True)  # noqa: S603
    #             except Exception as e:
    #                 print(f"Debugging off SLURM cluster: \n-----------------------------\n{e}\n-----------------------------\n")

    def status(
        self,
        *,
        count: list[str] | None = None,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        if include is not None:
            selected_fields = include
        else:
            selected_fields = self.items[0].dataclass_fields().keys()

        if exclude is not None:
            selected_fields = [f for f in selected_fields if f not in exclude]

        _df = pd.DataFrame.from_records(
            [{**item._values(), "status": item.status()} for item in self.items],
        )
        _df = _df[[*selected_fields, "status"]]
        unique_counts_per_selected = _df[selected_fields].nunique()
        sorted_fields = unique_counts_per_selected.sort_values(ascending=True)
        _df = _df.set_index(list(sorted_fields.index))

        if count is not None:
            sorted_fields = unique_counts_per_selected.sort_values(ascending=True)
            sorted_fields = sorted_fields.drop(index=count)
            return _df.reset_index().pivot_table(
                index=list(sorted_fields.index),
                columns="status",
                values=count,
                fill_value=0,
                aggfunc="count",
            )
        return _df

    def reset(self) -> None:
        for item in self.items:
            item.reset()
