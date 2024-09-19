import sys
from argparse import ArgumentParser
from pathlib import Path

import tomli

parser = ArgumentParser()
parser.add_argument(
    "--nightly",
    default="",
    help="List of dependencies to install from main branch for nightly tests, "
    "separated by commas.",
)
args = parser.parse_args()

CUSTOM_AUTO_SEPARATOR = """
# --- END OF CUSTOM SECTION ---
# The following was generated by 'tox -e deps', DO NOT EDIT MANUALLY!
"""


def write_dependencies(dependency_name: str, dependencies: list[str]) -> None:
    path = Path(f"{dependency_name}.in")
    if path.exists():
        sections = path.read_text().split(CUSTOM_AUTO_SEPARATOR)
        if len(sections) > 1:
            custom = sections[0]
        else:
            custom = ""
    else:
        custom = ""
    with path.open("w") as f:
        f.write(custom)
        f.write(CUSTOM_AUTO_SEPARATOR)
        f.write("\n".join(dependencies))
        f.write("\n")


with open("../pyproject.toml", "rb") as toml_file:
    pyproject = tomli.load(toml_file)
    dependencies = pyproject["project"].get("dependencies")
    if dependencies is None:
        raise RuntimeError("No dependencies found in pyproject.toml")
    dependencies = [dep.strip().strip('"') for dep in dependencies]
    test_dependencies = (
        pyproject["project"].get("optional-dependencies", {}).get("test", [])
    )
    test_dependencies = [dep.strip().strip('"') for dep in test_dependencies]


write_dependencies("base", dependencies)
write_dependencies("basetest", test_dependencies)


def as_nightly(repo: str) -> str:
    if "/" in repo:
        org, repo = repo.split("/")
    else:
        org = "scipp"
    if repo == "scipp":
        version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        base = "https://github.com/scipp/scipp/releases/download/nightly/scipp-nightly"
        suffix = "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        prefix = "scipp @ "
        return prefix + "-".join([base, version, version, suffix])
    return f"{repo} @ git+https://github.com/{org}/{repo}@main"


nightly = tuple(args.nightly.split(",") if args.nightly else [])
nightly_dependencies = [dep for dep in dependencies if not dep.startswith(nightly)]
nightly_dependencies += [as_nightly(arg) for arg in nightly]

write_dependencies("nightly", nightly_dependencies)
