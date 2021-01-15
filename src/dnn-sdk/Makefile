include ../kaldi.mk
CXX = g++
CXXFLAGS += -std=c++11 $(INCLUDE)\
           $(MCRO) $(EXTRA) 
ifeq ($(CUDA), true)
    RFLAGS += $(CUDA_LDFLAGS) $(CUDA_LDLIBS) 
endif 
OBJDIR = .
ADDLIBS =
LIBNAME = libdnn
LIBFILE = $(LIBNAME).dylib
CPPSOURCE = $(wildcard *.cpp)
OBJFILES = $(patsubst %cpp, $(OBJDIR)/%o, $(CPPSOURCE))
CUSOURCE = $(wildcard *.cu)
ifeq ($(CUDA), true)
    OB = $(patsubst %cu, $(OBJDIR)/%o, $(CUSOURCE))
    OBJFILES += $(OB)
endif
all: $(LIBFILE) $(CPPSOURCE)

$(LIBFILE): $(OBJFILES)
	ar -cru $(LIBFILE) $(OBJFILES)
	ranlib $(LIBFILE)

$(OBJDIR)/%.o:%.cpp
	         $(CXX) $(CXXFLAGS) -c -o $@ $< $(RFLAGS)
ifeq ($(CUDA), true)
    $(OBJDIR)/%.o:%.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -DUSE_CUDA -I../
endif
.PHONY: clean
clean:
	-rm $(OBJFILES) $(LIBFILE)
