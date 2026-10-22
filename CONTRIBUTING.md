# 参与 XU packages 开发

请遵守 [XU GitHub 仓库管理规范](https://github.com/xulang-org/xulang-org/blob/main/.github/REPOSITORY_RULES.md)。日常任务从 `dev` 建分支并通过 PR 合回；正式交付进入 `main`，后台设置按 [配置清单](https://github.com/xulang-org/xulang-org/blob/main/.github/GITHUB_SETTINGS.md) 另行验收。

本仓库使用根模块 `github.com/xulang-org/packages`，驱动目录是内部包，不新增独立 `xu.xun`。根模块的 `version:` 与 `xu:` 分别表示模块版本和选定 SDK；修改根模块内容，包括根贡献指南或模板时，同一批交付递增一次模块版本，不修改编译器产品版本。

## 修改与验证

在仓库根执行。`XU_ROOT` 可指定产品编译器 checkout，默认使用相邻 `../xu`；先按 README 准备本机测试依赖以及本次驱动所需服务。

```sh
python3 scripts/module_versions.py --bump
python3 scripts/module_versions.py --check
make test
```

`--bump` 前确认本批是否已经递增；新增兼容功能或不兼容接口变化时按规范选用相应版本级别。仅修改治理文件时运行版本与文档检查，无须启动无关数据库服务。修改 SQLite 原生桥时补充 `make native` 和相关验证，产物保持忽略。

## 提交 PR

使用本仓库 Issue、PR 模板，说明驱动或包的行为、兼容性、模块版本与 SDK 选择、实际服务环境、验证结果及回滚方法。未运行的外部服务或平台须列明，不能用归档引导实现替代产品编译器。

逐项审查暂存路径，并在提交前运行 `python3 scripts/check_git_hygiene.py --staged`。本机文档、测试报告、凭据、服务数据与生成原生库不进入源码提交。
