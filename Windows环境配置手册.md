# Windows 环境配置手册

> 本文档汇总了在 Windows 系统上配置 Node.js、Claude Code 代理及 CC Switch 的完整步骤，方便在新电脑上复现。

---

## 一、Node.js 安装（免安装版添加到系统 PATH）

### 1.1 准备 Node.js 免安装包

假设已下载并解压 Node.js Windows 免安装版到如下路径：

```
C:\Users\<你的用户名>\Downloads\node-v25.9.0-win-x64\node-v25.9.0-win-x64
```

### 1.2 添加到系统环境变量

以管理员身份打开 **PowerShell**，执行以下命令：

```powershell
$targetPath = "C:\Users\<你的用户名>\Downloads\node-v25.9.0-win-x64\node-v25.9.0-win-x64"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

# 检查是否已在 PATH 中
if ($currentPath -split ';' -contains $targetPath) {
    Write-Host "Already in PATH"
} else {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$targetPath", "Machine")
    Write-Host "Node.js path added to system PATH"
}
```

> **注意**：将 `<你的用户名>` 替换为实际 Windows 用户名。

### 1.3 验证安装

**关闭并重新打开**终端窗口，执行：

```bash
node --version
npm --version
```

预期输出：
```
v25.9.0
11.12.1
```

---

## 二、Claude Code 安装（npm 全局安装）

### 2.1 安装命令

在 Git Bash 或任意终端执行：

```bash
npm install -g @anthropic-ai/claude-code
```

### 2.2 常见问题：`claude: command not found`

#### 现象
`npm install -g` 显示安装成功，但在 Git Bash 中运行 `claude` 时提示 `command not found`。

#### 原因
npm 全局安装包的可执行文件存放在 `C:\Users\<用户名>\AppData\Roaming\npm` 目录下，而系统 `PATH` 中只配置了 Node.js 本体目录，缺少该 npm 全局目录。

#### 解决方案
以管理员身份打开 **PowerShell**，将 npm 全局目录添加到系统 PATH：

```powershell
$npmGlobalPath = "C:\Users\<你的用户名>\AppData\Roaming\npm"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

if ($currentPath -split ';' -contains $npmGlobalPath) {
    Write-Host "Already in PATH"
} else {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$npmGlobalPath", "Machine")
    Write-Host "Added npm global path to system PATH"
}
```

> **注意**：将 `<你的用户名>` 替换为实际 Windows 用户名。

保存后，**关闭并重新打开** Git Bash 窗口，再次执行 `claude` 即可。

---

## 三、Claude Code 代理配置（Git Bash + Clash）

### 2.1 前提条件

- 已安装 **Git Bash**
- 本地已运行 **Clash**（或其他代理工具），记住本地代理端口（本示例为 `7897`）

### 2.2 配置 Git Bash 代理（永久生效）

打开 Git Bash，执行以下命令创建/修改 `~/.bashrc`：

```bash
cat >> ~/.bashrc << 'EOF'
# Clash Proxy for Claude Code
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
EOF
```

> **注意**：将 `7897` 替换为你本地 Clash 实际监听的 HTTP 代理端口。

### 2.3 验证代理

**关闭并重新打开** Git Bash 窗口，执行：

```bash
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

预期输出：
```
http://127.0.0.1:7897
http://127.0.0.1:7897
```

### 2.4 启动 Claude Code

```bash
claude
```

---

## 三、Claude Code 跳过 Onboarding 引导

### 3.1 修改配置文件

Claude Code 首次运行会强制进入登录引导界面。若希望跳过此流程，需在以下两个配置文件中添加 `"hasCompletedOnboarding": true`。

#### 文件 1：`~/.claude.json`

找到用户主目录下的 `.claude.json` 文件（`C:\Users\<你的用户名>\.claude.json`），在最后一行前添加配置项：

```json
{
  "firstStartTime": "...",
  "opusProMigrationComplete": true,
  "sonnet1m45MigrationComplete": true,
  "seenNotifications": {},
  "migrationVersion": 13,
  "userID": "...",
  "changelogLastFetched": 1777686795130,
  "hasCompletedOnboarding": true
}
```

> 若文件不存在，先运行一次 `claude` 命令让其自动生成，再修改。

#### 文件 2：`~/.claude/settings.json`

找到或创建 `C:\Users\<你的用户名>\.claude\settings.json`，添加：

```json
{
  "theme": "dark",
  "hasCompletedOnboarding": true
}
```

### 3.2 生效

保存后，重新打开 Git Bash 并执行 `claude`，即可直接进入 Claude Code 主界面。

---

## 四、下载并安装 CC Switch

### 4.1 下载地址

GitHub Release：
- `https://github.com/farion1231/cc-switch/releases/tag/v3.14.1`

Windows 推荐下载 **MSI 安装版**：
- `CC-Switch-v3.14.1-Windows.msi`

直接下载链接：
```
https://github.com/farion1231/cc-switch/releases/download/v3.14.1/CC-Switch-v3.14.1-Windows.msi
```

### 4.2 命令行下载（可选）

在 PowerShell 中执行：

```powershell
$url = "https://github.com/farion1231/cc-switch/releases/download/v3.14.1/CC-Switch-v3.14.1-Windows.msi"
$output = "$env:USERPROFILE\Downloads\CC-Switch-v3.14.1-Windows.msi"
Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
```

### 4.3 安装

**方式一：图形界面安装**

双击下载的 `.msi` 文件，按安装向导完成即可。

**方式二：命令行静默安装**

以管理员身份运行 CMD：

```cmd
msiexec /i "C:\Users\%USERNAME%\Downloads\CC-Switch-v3.14.1-Windows.msi" /qn
```

---

## 五、常见问题速查

| 问题 | 解决方案 |
|------|----------|
| `node` 命令找不到 | 检查系统 PATH 是否包含 Node.js 目录，重启终端 |
| `npm` 在 PowerShell 报错 | PowerShell 执行策略限制，改用 CMD 或 `npm.cmd` |
| `claude` 命令找不到（Git Bash） | npm 全局目录未加入 PATH，参考本文档 **2.2** 节修复 |
| Claude Code 无法连接 | 确认 Clash 已运行，且 `~/.bashrc` 中的端口号正确 |
| Git Bash 代理未生效 | 确保修改的是 `~/.bashrc` 而非 `.bash_profile`，并重启 Git Bash |
| `.claude.json` 不存在 | 先运行一次 `claude` 命令，让程序自动生成配置文件 |

---

## 六、关键路径汇总

| 项目 | 路径 |
|------|------|
| Node.js 免安装目录 | `C:\Users\<用户名>\Downloads\node-v25.9.0-win-x64\node-v25.9.0-win-x64` |
| Git Bash 代理配置 | `C:\Users\<用户名>\.bashrc` |
| npm 全局可执行文件目录 | `C:\Users\<用户名>\AppData\Roaming\npm` |
| Claude Code 主配置 | `C:\Users\<用户名>\.claude.json` |
| Claude Code 设置 | `C:\Users\<用户名>\.claude\settings.json` |
| CC Switch 安装包 | `C:\Users\<用户名>\Downloads\CC-Switch-v3.14.1-Windows.msi` |
