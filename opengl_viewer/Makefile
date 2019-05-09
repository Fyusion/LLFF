UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	FLAGS = -lglfw -lGL -lGLU -ldl -lm -lXcursor -lXrandr -lXinerama -lXi -lX11 -lXxf86vm -lpthread
endif
ifeq ($(UNAME), Darwin)
	FLAGS = -framework OpenGl -framework Cocoa -framework IOKit -framework CoreVideo -lglfw3 
endif

opengl_viewer: glad.c viewer.cpp
	g++ glad.c viewer.cpp -o opengl_viewer -I. -std=c++11 $(FLAGS)

clean:
	rm opengl_viewer