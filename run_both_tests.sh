g++ test.cpp  -L /usr/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_dnn -lopencv_ml 
echo [INFO] Compiled.
echo [INFO] Running C++ version.
./a.out > logcpp.txt
echo [INFO] Running Python version.
python3 test.py > logpy.txt
echo [INFO] Done.