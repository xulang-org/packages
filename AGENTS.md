# Repository-local development rules

Before editing, read the selected compiler checkout's local
`docs/DEVELOPMENT.md`: use `XU_ROOT` when set, otherwise `../xu`.
This repository owns the external modules; Xu compiler and standard library
sources remain in the separate compiler repository.

Do not commit or push without the user's instruction. Stage explicit paths only.
Keep docs, reports, notes, logs, caches, generated native libraries and binaries
untracked. Do not move local compiler docs into this repository or CI.
Run `python3 scripts/check_git_hygiene.py --staged` before any commit.

The repository has one root module `github.com/xulang-org/packages`; driver
directories are its packages and must not gain separate xu.xun manifests.
Increment root module version once per batch; new module starts at 0.1.0.
The root xu: records the selected Xu compiler release and is independent of
the module version. Xu product VERSION remains owned by the compiler checkout.
Run `python3 scripts/module_versions.py --bump` followed by `--check`.
Run `make test` with the product compiler; never substitute the bootstrap oracle.
