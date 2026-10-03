@echo off
rem Rebuild the CPU kernels (prebuilt DLLs are included; only needed after editing kernels\*.c).
rem Both gptoss_cpu_*.dll read raw expert stores by default and packed-scale stores after gptoss_set_scale_layout(1)
rem (server.py sets it from the store); a build made before that switch cannot read a packed store.
rem Needs a MinGW-w64 gcc with OpenMP on PATH (e.g. w64devkit or MSYS2 mingw64).
cd /d "%~dp0.."
gcc -O3 -march=native -fopenmp -shared -o gptoss_cpu_cap2.dll kernels\gptoss_cpu_cap2.c || exit /b 1
gcc -O3 -march=native -fopenmp -shared -o gptoss_cpu_multi.dll kernels\gptoss_cpu_multi.c || exit /b 1
gcc -O3 -march=native -fopenmp -shared -o memcpy_mt.dll kernels\memcpy_mt.c || exit /b 1
gcc -O3 -march=native -fopenmp -shared -o membw_probe.dll kernels\membw_probe.c || exit /b 1
echo built gptoss_cpu_cap2.dll gptoss_cpu_multi.dll memcpy_mt.dll membw_probe.dll
