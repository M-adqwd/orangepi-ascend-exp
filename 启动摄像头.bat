@echo off
chcp 65001 >nul
setlocal
set HOST=192.168.8.199
set USER=HwHiAiUser
set PORT=8000
set URL=http://%HOST%:%PORT%

echo ==================================================
echo   Orange Pi 摄像头实时检测  --  启动
echo ==================================================
echo.
echo  目标: %USER%@%HOST%
echo  提示输入密码时，密码是 Mind@123
echo  （输入过程中屏幕不显示字符，属正常现象）
echo.
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 %USER%@%HOST% "bash ~/exp/exp3/start_stream.sh %PORT%"
if errorlevel 1 goto FAIL

echo.
echo  等待服务就绪...
timeout /t 9 /nobreak >nul

curl -s -m 6 -o nul "%URL%/stats" && goto OK
echo.
echo  [警告] 服务暂未响应，可能模型还在加载。
echo         浏览器打开后若白屏，等 10 秒刷新一次即可。
goto OPEN

:OK
echo  [OK] 服务正常，实时检测已在运行。

:OPEN
echo.
start "" "%URL%"
echo  已在浏览器打开: %URL%
echo.
echo  实时画面   %URL%
echo  单帧快照   %URL%/snapshot
echo  运行状态   %URL%/stats
echo.
echo  关闭服务请双击「停止摄像头.bat」
echo.
pause
exit /b 0

:FAIL
echo.
echo  [失败] 无法连接板子或启动失败。请检查：
echo    1. 板子是否已开机（等 1-2 分钟完全启动）
echo    2. 网线是否插好 / 是否同一 WiFi
echo    3. 在命令行执行 ping %HOST% 看是否通
echo.
pause
exit /b 1
