# ============================================================================
#  LinuxGUI.ps1  --  WSLg 图形启动器（Windows 侧，带窗口界面）
#
#  双击 Linux-GUI.cmd 会打开这个窗口，点按钮就能在 Windows 桌面上启动
#  WSL 里的图形程序，或者一整个 XFCE 桌面。
#
#  命令行也兼容旧习惯：
#     Linux-GUI.cmd                  打开图形界面
#     Linux-GUI.cmd desktop [宽x高]   直接启动完整桌面
#     Linux-GUI.cmd stop             关闭完整桌面
#     Linux-GUI.cmd check            环境自检
#     Linux-GUI.cmd fix              重启 WSL 后重开桌面
#     Linux-GUI.cmd <命令> [参数]     启动单个程序（thunar / firefox ...）
#
#  为什么需要它：
#    WSLg 有时会“Linux 侧一切正常，但 Windows 死活不显示窗口”，而且没有任何
#    报错。只有在 Windows 侧枚举 msrdc 的 RAIL_WINDOW 才能发现。本脚本在启动
#    后会主动检测，一旦没检测到就自动 wsl --shutdown 再重试一次。
# ============================================================================

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

if (-not ('WslgRail' -as [type])) {
  Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class WslgRail {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr l);
  [DllImport("user32.dll")] public static extern int  GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern int  GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
}
'@
}

# ---------------------------------------------------------------------------
#  基础：发行版 / 后端 / 窗口检测
# ---------------------------------------------------------------------------
$script:Distro = $null

function Get-Distro {
  $raw = & wsl.exe -l -q 2>$null
  foreach ($line in @($raw)) {
    $n = (($line -replace "`0", '') -replace "`r", '').Trim()
    if ($n) { return $n }
  }
  return $null
}

function Test-Backend {
  if (-not $script:Distro) { return $false }
  & wsl.exe -d $script:Distro -e bash -lc 'test -x "$HOME/.local/bin/linux-gui"' 2>$null
  return ($LASTEXITCODE -eq 0)
}

# 返回当前可见的 WSLg 窗口标题；$Filter 为空表示全部
function Get-RailWindows {
  param([string]$Filter = '')
  $script:railHits = New-Object System.Collections.ArrayList
  $cb = [WslgRail+EnumProc]{
    param($h, $l)
    $cls = New-Object System.Text.StringBuilder 256
    [WslgRail]::GetClassName($h, $cls, 256) | Out-Null
    if ($cls.ToString() -eq 'RAIL_WINDOW' -and [WslgRail]::IsWindowVisible($h)) {
      $wpid = 0
      [WslgRail]::GetWindowThreadProcessId($h, [ref]$wpid) | Out-Null
      $pn = (Get-Process -Id $wpid -ErrorAction SilentlyContinue).ProcessName
      if ($pn -eq 'msrdc') {
        $sb = New-Object System.Text.StringBuilder 512
        [WslgRail]::GetWindowText($h, $sb, 512) | Out-Null
        $t = $sb.ToString()
        if ([string]::IsNullOrEmpty($Filter) -or $t -like "*$Filter*") {
          [void]$script:railHits.Add($t)
        }
      }
    }
    return $true
  }
  [WslgRail]::EnumWindows($cb, [IntPtr]::Zero) | Out-Null
  return $script:railHits
}

# ---------------------------------------------------------------------------
#  构造发送给 WSL 的命令
# ---------------------------------------------------------------------------
function Quote-BashArg([string]$s) {
  if ($s -match "^[A-Za-z0-9_./:@%+=,-]+$") { return $s }
  return "'" + ($s -replace "'", "'\''") + "'"
}

function New-BackendCmd([string[]]$AppArgs) {
  $q = (@($AppArgs) | ForEach-Object { Quote-BashArg $_ }) -join ' '
  return 'exec "$HOME/.local/bin/linux-gui" ' + $q
}

function Invoke-WslSync([string]$Bash) {
  $out = & wsl.exe -d $script:Distro -e bash -lc $Bash 2>&1 | Out-String
  return [pscustomobject]@{ Output = $out; Code = $LASTEXITCODE }
}

# ---------------------------------------------------------------------------
#  命令行模式（不开窗口，输出到控制台）
# ---------------------------------------------------------------------------
function Invoke-CliAction([string[]]$A) {
  try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
  $script:Distro = Get-Distro
  if (-not $script:Distro) {
    Write-Host '[XX] 找不到 WSL 发行版，请先执行: wsl --install' -ForegroundColor Red
    return 1
  }
  if (-not (Test-Backend)) {
    Write-Host '[XX] WSL 里找不到 ~/.local/bin/linux-gui' -ForegroundColor Red
    return 1
  }

  $act = if ($A.Count -gt 0 -and $A[0]) { $A[0].ToLower() } else { '' }

  # 兼容旧写法：--check / -c / restart 等都能用
  $norm = $act.TrimStart('-', '/')
  if ($norm -eq 'stop-desktop') { $norm = 'stop' }
  switch ($norm) {
    'desktop' {
      $geom = if ($A.Count -gt 1 -and $A[1]) { $A[1] } else { '1600x900' }
      return (Invoke-CliDesktop $geom)
    }
    'stop' {
      $r = Invoke-WslSync (New-BackendCmd @('--stop-desktop'))
      Write-Host $r.Output
      return $r.Code
    }
    'check' {
      $r = Invoke-WslSync (New-BackendCmd @('--check'))
      Write-Host $r.Output
      return $r.Code
    }
    'list' {
      $r = Invoke-WslSync (New-BackendCmd @('--list'))
      Write-Host $r.Output
      return $r.Code
    }
    'fix' { return (Invoke-CliDesktop '1600x900') }
    'restart' {
      Write-Host '正在重启 WSL ...'
      & wsl.exe --shutdown 2>&1 | Out-Null
      Start-Sleep -Seconds 2
      return (Invoke-CliDesktop '1600x900')
    }
    'help' { Get-CliUsage; return 0 }
    default {
      $r = Invoke-WslSync (New-BackendCmd $A)
      Write-Host $r.Output
      return $r.Code
    }
  }
}

function Get-CliUsage {
  Write-Host @'
Linux GUI 启动器 (WSLg) 命令行用法:

  Linux-GUI.cmd                  打开图形界面（推荐，直接双击）
  Linux-GUI.cmd desktop [WxH]    启动完整 XFCE 桌面
  Linux-GUI.cmd stop             关闭完整桌面
  Linux-GUI.cmd check            环境自检
  Linux-GUI.cmd fix              重启 WSL 后重开桌面
  Linux-GUI.cmd <命令> [参数]     启动单个程序，例如 thunar / firefox
'@
}

function Invoke-CliDesktop([string]$Geom) {
  Write-Host "正在启动完整 XFCE 桌面 ($Geom) ..."
  $r = Invoke-WslSync (New-BackendCmd @('--desktop', $Geom))
  Write-Host $r.Output
  if ($r.Code -ne 0) { return $r.Code }

  Write-Host '正在确认 Windows 是否真的显示了窗口 ...'
  for ($i = 0; $i -lt 20; $i++) {
    if ((Get-RailWindows 'XFCE Desktop').Count -gt 0) {
      Write-Host '[OK] 已显示窗口 "XFCE Desktop"。' -ForegroundColor Green
      return 0
    }
    Start-Sleep -Milliseconds 500
  }

  Write-Host '[!!] 没等到窗口，WSLg 会话可能失效。正在自动重启 WSL 后重试 ...' -ForegroundColor Yellow
  & wsl.exe --shutdown 2>&1 | Out-Null
  Start-Sleep -Seconds 2
  $r = Invoke-WslSync (New-BackendCmd @('--desktop', $Geom))
  Write-Host $r.Output
  if ($r.Code -ne 0) { return $r.Code }

  for ($i = 0; $i -lt 20; $i++) {
    if ((Get-RailWindows 'XFCE Desktop').Count -gt 0) {
      Write-Host '[OK] 重试成功，已显示窗口 "XFCE Desktop"。' -ForegroundColor Green
      return 0
    }
    Start-Sleep -Milliseconds 500
  }
  Write-Host '[XX] 仍然没有窗口。请在 PowerShell 执行: wsl --shutdown  然后重试。' -ForegroundColor Red
  return 1
}

# ===========================================================================
#  下面是图形界面部分
# ===========================================================================
$ColorOk   = [System.Drawing.Color]::FromArgb(0x1B, 0x7F, 0x3B)
$ColorErr  = [System.Drawing.Color]::FromArgb(0xB0, 0x1B, 0x1B)
$ColorBusy = [System.Drawing.Color]::FromArgb(0x1F, 0x5F, 0xA8)
$ColorWarn = [System.Drawing.Color]::FromArgb(0xB0, 0x6A, 0x00)

$script:Job            = $null
$script:Busy           = $false
$script:PendingVerify  = $null
$script:VerifyRetryCmd = $null
$script:VerifyRetried  = $false
$script:VerifyTicks    = 0

# ---- 控件 ----------------------------------------------------------------
$form = New-Object System.Windows.Forms.Form
$form.Text = 'Linux GUI 启动器 (WSLg)'
$form.ClientSize = New-Object System.Drawing.Size(688, 560)
$form.StartPosition = 'CenterScreen'
$form.MinimumSize = New-Object System.Drawing.Size(620, 460)
$form.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', 9)
$form.BackColor = [System.Drawing.Color]::FromArgb(0xF5, 0xF6, 0xF8)

$title = New-Object System.Windows.Forms.Label
$title.Text = '在 Windows 上运行 Linux 图形程序'
$title.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', 12, [System.Drawing.FontStyle]::Bold)
$title.Location = New-Object System.Drawing.Point(16, 12)
$title.Size = New-Object System.Drawing.Size(420, 26)
$form.Controls.Add($title)

$lblDistro = New-Object System.Windows.Forms.Label
$lblDistro.Text = '正在检查环境 ...'
$lblDistro.Location = New-Object System.Drawing.Point(18, 42)
$lblDistro.Size = New-Object System.Drawing.Size(654, 18)
$lblDistro.ForeColor = [System.Drawing.Color]::FromArgb(0x55, 0x55, 0x55)
$form.Controls.Add($lblDistro)

function New-Button([string]$Text, [int]$X, [int]$Y, [int]$W, [int]$H, $Handler) {
  $b = New-Object System.Windows.Forms.Button
  $b.Text = $Text
  $b.Location = New-Object System.Drawing.Point($X, $Y)
  $b.Size = New-Object System.Drawing.Size($W, $H)
  $b.UseVisualStyleBackColor = $true
  $b.FlatStyle = 'System'
  $b.Add_Click($Handler)
  $form.Controls.Add($b)
  return $b
}

# ---- 动作 ----------------------------------------------------------------
function Set-Status([string]$Text, $Color) {
  $script:status.Text = $Text
  if ($Color) { $script:status.ForeColor = $Color }
}

function Add-Log([string]$Text) {
  if (-not $script:log) { return }
  $t = ($Text -replace "`r`n", "`n") -replace "`r", "`n"
  $script:log.AppendText($t + "`r`n")
  $script:log.SelectionStart = $script:log.TextLength
  $script:log.ScrollToCaret()
  [System.Windows.Forms.Application]::DoEvents()
}

function Set-BusyUI([bool]$Busy) {
  foreach ($b in @($script:btnDesktop, $script:btnStop, $script:btnTerm, $script:btnFiles,
                   $script:btnFirefox, $script:btnCheck, $script:btnFix, $script:btnLogs)) {
    if ($b) { $b.Enabled = -not $Busy }
  }
  $script:progress.Style = if ($Busy) { 'Marquee' } else { 'Blocks' }
  if ($Busy) { $script:progress.MarqueeAnimationSpeed = 30 } else { $script:progress.MarqueeAnimationSpeed = 0 }
}

function Done-Busy {
  $script:Busy = $false
  $script:PendingVerify = $null
  $script:VerifyRetryCmd = $null
  Set-BusyUI $false
}

function Start-WslJob {
  param(
    [string]$Label,
    [string]$Bash,
    [string]$Verify = '',
    [string]$RetryBash = '',
    [bool]$NoAutoRetry = $false
  )
  if ($script:Busy) { return }
  $script:Busy = $true
  Set-BusyUI $true
  Set-Status "$Label ..." $ColorBusy
  Add-Log "> $Label"

  $script:PendingVerify  = if ($Verify) { $Verify } else { $null }
  $script:VerifyRetryCmd = if ($RetryBash) { $RetryBash } else { $null }
  $script:VerifyRetried  = $NoAutoRetry
  $script:VerifyTicks    = 0

  $script:Job = Start-Job -ScriptBlock {
    param($d, $b)
    $o = & wsl.exe -d $d -e bash -lc $b 2>&1 | Out-String
    [pscustomobject]@{ Output = $o; Code = $LASTEXITCODE }
  } -ArgumentList $script:Distro, $Bash

  $script:PollTimer.Start()
}

function Start-DesktopJob([string]$Geom, [bool]$NoAutoRetry) {
  $bash = New-BackendCmd @('--desktop', $Geom)
  Start-WslJob -Label "启动完整桌面 ($Geom)" -Bash $bash -Verify 'XFCE Desktop' -RetryBash $bash -NoAutoRetry $NoAutoRetry
}

function Restart-Desktop {
  if ($script:Busy) { return }
  $script:Busy = $true
  Set-BusyUI $true
  Set-Status '正在重启 WSL（修复 WSLg）...' $ColorWarn
  Add-Log '> 重启 WSL 并重新打开桌面'
  try { & wsl.exe --shutdown 2>&1 | Out-Null } catch { }
  Start-Sleep -Seconds 2
  Add-Log '  WSL 已停止，正在重新启动桌面 ...'
  $script:Busy = $false
  Start-DesktopJob $script:comboGeom.Text $true
}

# ---- 按钮 -----------------------------------------------------------------
$script:comboGeom = New-Object System.Windows.Forms.ComboBox
$script:comboGeom.DropDownStyle = 'DropDown'
$script:comboGeom.Location = New-Object System.Drawing.Point(560, 70)
$script:comboGeom.Size = New-Object System.Drawing.Size(110, 24)
[void]$script:comboGeom.Items.AddRange(@('1280x800', '1440x900', '1600x900', '1920x1080', '2560x1440'))
$script:comboGeom.Text = '1600x900'
$form.Controls.Add($script:comboGeom)

$lblGeom = New-Object System.Windows.Forms.Label
$lblGeom.Text = '桌面分辨率'
$lblGeom.Location = New-Object System.Drawing.Point(480, 74)
$lblGeom.Size = New-Object System.Drawing.Size(76, 18)
$form.Controls.Add($lblGeom)

$script:btnDesktop = New-Button '启动完整桌面' 16 70 180 36 {
  Start-DesktopJob $script:comboGeom.Text $false
}
$script:btnStop    = New-Button '关闭完整桌面' 206 70 130 36 {
  Start-WslJob -Label '关闭完整桌面' -Bash (New-BackendCmd @('--stop-desktop'))
}
$script:btnTerm    = New-Button '终端' 16 114 180 36 {
  Start-WslJob -Label '启动终端' -Bash (New-BackendCmd @('xfce4-terminal'))
}
$script:btnFiles   = New-Button '文件管理器' 206 114 130 36 {
  Start-WslJob -Label '启动文件管理器' -Bash (New-BackendCmd @('thunar'))
}
$script:btnFirefox = New-Button 'Firefox 浏览器' 16 158 180 36 {
  Start-WslJob -Label '启动 Firefox' -Bash (New-BackendCmd @('firefox'))
}
$script:btnCheck   = New-Button '环境自检' 206 158 130 36 {
  Start-WslJob -Label '环境自检' -Bash (New-BackendCmd @('--check'))
}
$script:btnFix     = New-Button '重启 WSL 修复' 16 202 180 36 {
  Restart-Desktop
}
$script:btnLogs    = New-Button '查看日志' 206 202 130 36 {
  $b = 'lg="${XDG_RUNTIME_DIR:-/tmp}/linux-gui-logs"; ' +
       'echo "== desktop.log =="; tail -n 30 "$lg/desktop.log" 2>/dev/null; ' +
       'echo; echo "== xephyr.log =="; tail -n 15 "$lg/xephyr.log" 2>/dev/null'
  Start-WslJob -Label '查看日志' -Bash $b
}

# ---- 状态 / 进度 / 日志 ---------------------------------------------------
$script:status = New-Object System.Windows.Forms.Label
$script:status.Text = '就绪'
$script:status.Location = New-Object System.Drawing.Point(18, 250)
$script:status.Size = New-Object System.Drawing.Size(654, 20)
$script:status.Anchor = 'Top,Left,Right'
$script:status.ForeColor = $ColorOk
$form.Controls.Add($script:status)

$script:progress = New-Object System.Windows.Forms.ProgressBar
$script:progress.Location = New-Object System.Drawing.Point(18, 274)
$script:progress.Size = New-Object System.Drawing.Size(654, 6)
$script:progress.Anchor = 'Top,Left,Right'
$script:progress.Style = 'Blocks'
$form.Controls.Add($script:progress)

$script:log = New-Object System.Windows.Forms.TextBox
$script:log.Location = New-Object System.Drawing.Point(18, 290)
$script:log.Size = New-Object System.Drawing.Size(654, 254)
$script:log.Anchor = 'Top,Bottom,Left,Right'
$script:log.Multiline = $true
$script:log.ScrollBars = 'Vertical'
$script:log.ReadOnly = $true
$script:log.WordWrap = $false
$script:log.BackColor = [System.Drawing.Color]::FromArgb(0x1E, 0x1E, 0x1E)
$script:log.ForeColor = [System.Drawing.Color]::FromArgb(0xE6, 0xE6, 0xE6)
$script:log.Font = New-Object System.Drawing.Font('Consolas', 9)
$form.Controls.Add($script:log)

# ---- 定时器 ---------------------------------------------------------------
$script:PollTimer = New-Object System.Windows.Forms.Timer
$script:PollTimer.Interval = 250
$script:PollTimer.Add_Tick({
  if (-not $script:Job) { $script:PollTimer.Stop(); return }
  if ($script:Job.State -eq 'Running' -or $script:Job.State -eq 'NotStarted') { return }

  $script:PollTimer.Stop()
  $received = Receive-Job $script:Job 2>&1
  Remove-Job $script:Job -Force
  $script:Job = $null

  $code = $null
  $text = ''
  foreach ($r in $received) {
    if ($r -is [pscustomobject] -and ($r.PSObject.Properties.Name -contains 'Code')) {
      $code = $r.Code; $text += $r.Output
    } else {
      $text += "$r`r`n"
    }
  }
  if ($text.Trim()) { Add-Log $text.TrimEnd() }

  if ($code -ne 0) {
    Set-Status "失败（退出码 $code）" $ColorErr
    Done-Busy
    return
  }

  if ($script:PendingVerify) {
    $script:VerifyTicks = 0
    $script:VerifyTimer.Start()
  } else {
    Set-Status '完成' $ColorOk
    Done-Busy
  }
})

$script:VerifyTimer = New-Object System.Windows.Forms.Timer
$script:VerifyTimer.Interval = 500
$script:VerifyTimer.Add_Tick({
  $script:VerifyTicks++
  $hits = Get-RailWindows $script:PendingVerify
  if ($hits.Count -gt 0) {
    $script:VerifyTimer.Stop()
    Add-Log "[OK] Windows 已显示窗口：$($hits[0])"
    Set-Status "已显示：$($hits[0])" $ColorOk
    Done-Busy
    return
  }
  if ($script:VerifyTicks -ge 20) {
    $script:VerifyTimer.Stop()
    if (-not $script:VerifyRetried -and $script:VerifyRetryCmd) {
      $script:VerifyRetried = $true
      Add-Log '[!!] 10 秒内没有等到窗口，WSLg 会话很可能已经失效。'
      Add-Log '     自动执行 wsl --shutdown 修复，然后重试 ...'
      Set-Status '正在重启 WSL 修复 ...' $ColorWarn
      try { & wsl.exe --shutdown 2>&1 | Out-Null } catch { }
      Start-Sleep -Seconds 2
      Add-Log '     重新启动桌面 ...'
      $script:Job = Start-Job -ScriptBlock {
        param($d, $b)
        $o = & wsl.exe -d $d -e bash -lc $b 2>&1 | Out-String
        [pscustomobject]@{ Output = $o; Code = $LASTEXITCODE }
      } -ArgumentList $script:Distro, $script:VerifyRetryCmd
      $script:VerifyTicks = 0
      $script:PollTimer.Start()
    } else {
      Set-Status '失败：Windows 上没有出现窗口' $ColorErr
      Add-Log '========================================================='
      Add-Log ' 启动流程跑完了，但 Windows 上始终没有出现窗口。'
      Add-Log ' 可尝试：点“环境自检”，或在 PowerShell 里执行 wsl --shutdown 后重试。'
      Add-Log '========================================================='
      Done-Busy
    }
  }
})

# ---- 启动时自检 -----------------------------------------------------------
$form.Add_Shown({
  Add-Log '正在检查运行环境 ...'
  Set-BusyUI $true
  Set-Status '正在检查 ...' $ColorBusy

  $script:Distro = Get-Distro
  if (-not $script:Distro) {
    $lblDistro.Text = '未检测到 WSL 发行版'
    Set-Status '没有检测到 WSL，请先在 PowerShell 执行: wsl --install' $ColorErr
    Add-Log '[XX] 找不到任何 WSL 发行版。'
    Set-BusyUI $false
    return
  }
  $lblDistro.Text = "发行版：$script:Distro"

  if (-not (Test-Backend)) {
    Set-Status 'WSL 中缺少启动脚本' $ColorErr
    Add-Log '[XX] WSL 里找不到 ~/.local/bin/linux-gui'
    Add-Log '     请确认该文件存在并具有可执行权限。'
    Set-BusyUI $false
    return
  }

  $visible = Get-RailWindows
  Add-Log "[OK] 环境就绪（$script:Distro）"
  if ($visible.Count -gt 0) {
    Add-Log ("     当前已有 {0} 个 WSLg 窗口：{1}" -f $visible.Count, (($visible) -join ' / '))
  } else {
    Add-Log '     当前没有 WSLg 窗口（正常，启动后会显示）。'
  }
  Set-Status '就绪' $ColorOk
  Set-BusyUI $false
})

# ---------------------------------------------------------------------------
#  入口
# ---------------------------------------------------------------------------
if ($args.Count -gt 0) {
  exit (Invoke-CliAction $args)
}

try {
  [void]$form.ShowDialog()
} catch {
  [System.Windows.Forms.MessageBox]::Show("启动器出错：`n`n$($_.Exception.Message)", 'Linux GUI 启动器',
      [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
  exit 1
}
