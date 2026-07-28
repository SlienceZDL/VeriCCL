# VeriCCL Python 3.13 与 dd 依赖兼容设计

## 目标

修复 Ubuntu 服务器使用 Python 3.13 执行 `pip install -e .` 时，
`dd 0.5.7` 源码构建因缺少 `pkg_resources` 而失败的问题。修复后，
VeriCCL 应明确支持 CPython 3.10、3.11、3.12 和 3.13，并保持现有
BDD 验证语义不变。

## 根因

`setup.py` 当前声明 `python_requires=">=3.9"`，但运行时依赖固定为
`dd>=0.5.7,<0.6`。`dd 0.5.7` 只为 Linux CPython 3.10 发布 wheel；
Python 3.13 因此回退到源码构建，并在最新隔离构建环境中导入已经移除
的 `pkg_resources`。README 未在创建虚拟环境前声明或检查支持版本，
导致错误发生在依赖构建阶段。

## 兼容策略

`setup.py` 将支持范围收敛为：

```text
>=3.10,<3.14
```

`dd` 使用环境标记按解释器版本解析：

```text
dd>=0.5.7,<0.6; python_version == "3.10"
dd>=0.6,<0.7; python_version >= "3.11"
```

Python 3.10 继续使用现有且已验证的 0.5.x API；Python 3.11–3.13 使用
提供对应 Linux wheel 的 0.6.x。VeriCCL 继续仅通过
`vericcl.verification.bdd_backend` 使用 `dd.autoref.BDD`，不改变 BDD
变量、集合运算、枚举、错误映射或验证结果。

## 安装说明

英文和中文 README 在创建虚拟环境前明确要求 CPython 3.10–3.13，并
增加相同的可执行版本检查。版本不受支持时，命令应立即输出实际版本和
支持范围，而不是进入依赖构建。

既有 SSH 与 HTTPS 克隆流程、虚拟环境目录、依赖安装顺序和后续
quickstart 命令保持不变。两份 README 的 Bash 代码块继续逐字一致。

## 测试与验证

自动化测试应验证：

1. `setup.py` 的 Python 范围与两个互斥的 `dd` 环境标记；
2. README 版本检查的真实执行行为，包括支持版本通过和不支持版本拒绝；
3. 两份 README 的 Bash 代码块一致；
4. 现有 BDD 单元测试和完整软件测试不回归。

交付前执行以下环境验证：

1. 使用 Python 3.13 新建隔离环境，执行项目可编辑安装和 BDD 测试；
2. 验证 Linux CPython 3.10 能解析 `dd 0.5.x` wheel；
3. 验证 Linux CPython 3.11、3.12、3.13 能解析 `dd 0.6.x` wheel；
4. 运行完整 `pytest`、`pip check`、`compileall` 和 `git diff --check`。

若当前主机不能执行某个平台验证，报告必须明确标为 `not_run`，不得以
静态元数据检查代替实际安装。

## 集成

修复在 `fix/python313-dd-compatibility` 分支完成。验证通过后创建并
合并 PR 到 `feature/vericcl-implementation`，随后更新本地主分支并
删除临时分支和 worktree。
