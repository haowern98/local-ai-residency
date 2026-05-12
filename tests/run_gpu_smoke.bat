@echo off
setlocal enabledelayedexpansion

if "%MOSAICVRAM_EXE%"=="" set "MOSAICVRAM_EXE=build-llama-onnx\mosaicvram.exe"
if "%PROMPT_FILE%"=="" set "PROMPT_FILE=tests\fixtures\15k_prompt.txt"
if "%DEVICE%"=="" set "DEVICE=0"
if "%THREADS%"=="" set "THREADS=8"
if "%LLAMA_CTX%"=="" set "LLAMA_CTX=16384"
if "%LLAMA_BATCH%"=="" set "LLAMA_BATCH=512"
if "%LLAMA_GPU_LAYERS%"=="" set "LLAMA_GPU_LAYERS=-1"
if "%ONNX_PREFILL_CHUNK%"=="" set "ONNX_PREFILL_CHUNK=512"
if "%ONNX_CONTROL_INPUT_DEVICE%"=="" set "ONNX_CONTROL_INPUT_DEVICE=cpu"

if "%1"=="" (
  echo Usage: tests\run_gpu_smoke.bat llama^|onnx^|mixed
  echo.
  echo Required for llama/mixed: LLAMA_MODEL
  echo Required for onnx/mixed: ONNX_MODEL and ONNX_TOKENIZER
  exit /b 2
)

set "MODE=%1"
set "GENERATED_DIR=tests\generated"
if not exist "%GENERATED_DIR%" mkdir "%GENERATED_DIR%"

if "%MODE%"=="llama" goto llama
if "%MODE%"=="onnx" goto onnx
if "%MODE%"=="mixed" goto mixed
echo Unknown mode: %MODE%
exit /b 2

:check_llama
if "%LLAMA_MODEL%"=="" (
  echo LLAMA_MODEL is required.
  exit /b 2
)
exit /b 0

:check_onnx
if "%ONNX_MODEL%"=="" (
  echo ONNX_MODEL is required.
  exit /b 2
)
if "%ONNX_TOKENIZER%"=="" (
  echo ONNX_TOKENIZER is required.
  exit /b 2
)
exit /b 0

:render_plan
set "TEMPLATE=%~1"
set "PLAN=%~2"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$text = Get-Content -LiteralPath '%TEMPLATE%' -Raw; " ^
  "$prompt = Get-Content -LiteralPath '%PROMPT_FILE%' -Raw; " ^
  "$text = $text.Replace('{{PROMPT_TEXT}}', $prompt.Trim()); " ^
  "$text = $text.Replace('{{PROMPT_FILE}}', '%PROMPT_FILE%'); " ^
  "$text = $text.Replace('{{LLAMA_MODEL}}', '%LLAMA_MODEL%'); " ^
  "$text = $text.Replace('{{LLAMA_CTX}}', '%LLAMA_CTX%'); " ^
  "$text = $text.Replace('{{LLAMA_BATCH}}', '%LLAMA_BATCH%'); " ^
  "$text = $text.Replace('{{LLAMA_GPU_LAYERS}}', '%LLAMA_GPU_LAYERS%'); " ^
  "$text = $text.Replace('{{ONNX_MODEL}}', '%ONNX_MODEL%'); " ^
  "$text = $text.Replace('{{ONNX_TOKENIZER}}', '%ONNX_TOKENIZER%'); " ^
  "$text = $text.Replace('{{ONNX_PREFILL_CHUNK}}', '%ONNX_PREFILL_CHUNK%'); " ^
  "$text = $text.Replace('{{ONNX_CONTROL_INPUT_DEVICE}}', '%ONNX_CONTROL_INPUT_DEVICE%'); " ^
  "$text = $text.Replace('{{DEVICE}}', '%DEVICE%'); " ^
  "$text = $text.Replace('{{THREADS}}', '%THREADS%'); " ^
  "Set-Content -LiteralPath '%PLAN%' -Value $text -Encoding ASCII"
if errorlevel 1 exit /b 1
exit /b 0

:llama
call :check_llama
if errorlevel 1 exit /b 1
set "PLAN=%GENERATED_DIR%\llama_long_context.plan"
call :render_plan tests\plans\templates\llama_long_context.plan "%PLAN%"
if errorlevel 1 exit /b 1
"%MOSAICVRAM_EXE%" residency-run --plan "%PLAN%"
exit /b %ERRORLEVEL%

:onnx
call :check_onnx
if errorlevel 1 exit /b 1
set "PLAN=%GENERATED_DIR%\onnx_long_context.plan"
call :render_plan tests\plans\templates\onnx_long_context.plan "%PLAN%"
if errorlevel 1 exit /b 1
"%MOSAICVRAM_EXE%" residency-run --plan "%PLAN%"
exit /b %ERRORLEVEL%

:mixed
call :check_llama
if errorlevel 1 exit /b 1
call :check_onnx
if errorlevel 1 exit /b 1
set "PLAN=%GENERATED_DIR%\mixed_backend.plan"
call :render_plan tests\plans\templates\mixed_backend.plan "%PLAN%"
if errorlevel 1 exit /b 1
"%MOSAICVRAM_EXE%" residency-run --plan "%PLAN%"
exit /b %ERRORLEVEL%
