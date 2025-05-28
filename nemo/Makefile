ifeq ($(CXX),)
        CXX= g++
endif

LIBDIR= $(NEMOOBJ)/acc

MAKECFLAGS:= $(CFLAGS) -I$(NEMOINC)
MAKELDFLAGS:= $(LDFLAGS) -L$(NEMOLIB)

LINKOPTIONS:= -shared -lm -lgsl -lgslcblas -lnemo

all: $(LIBDIR)/PowSphwCut.so

$(LIBDIR)/PowSphwCut.so: PowSphwCut.cc
	$(CXX) -o $@ $< \
	$(MAKECFLAGS) $(MAKELDFLAGS) $(LINKOPTIONS) \
	-march=native -Wall -Wno-unknown-pragmas -fPIC
