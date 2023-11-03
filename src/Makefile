INC_PATH=../include
LIB_PATH=../lib

LIBS = -linference_engine -lopencv_dnn -lopencv_imgproc -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_videoio -lngraph -linference_engine_transformations -ltbb

all: VehicleCounter.o

VehicleCounter.o:
	$(CXX) -std=c++14 -I$(INC_PATH) CentroidTracker.cpp main.cpp -o VehicleCounter -L$(LIB_PATH) $(LIBS) -Wl,--disable-new-dtags -Wl,-rpath=../lib
