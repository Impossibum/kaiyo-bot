@echo off
Rem count start = 0, incrementing value = 1, max value = 7
FOR /L %%x IN (0, 1, 7) DO (
	start python worker.py %%x
    	timeout 35 >nul
    	echo "started instance %%x"
	)

pause