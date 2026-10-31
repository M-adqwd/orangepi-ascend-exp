@echo off
chcp 65001 >nul
setlocal
set HOST=192.168.8.199
set USER=HwHiAiUser

echo ==================================================
echo   Orange Pi 摄像头实时检测  --  停止
echo ==================================================
echo.
echo  提示输入密码时，密码是 Mind@123
echo.
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 %USER%@%HOST% "bash ~/exp/exp3/stop_stream.sh"
if errorlevel 1 (
  echo.
  echo  [失败] 未能连接板子，服务可能仍在运行。
  pause
  exit /b 1
)
echo.
echo  已停止。浏览器里的画面会在几秒内冻结，属正常。
echo.
pause
