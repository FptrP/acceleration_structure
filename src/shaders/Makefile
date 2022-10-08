VERT_SOURCES = $(wildcard */*.vert)
FRAG_SOURCES = $(wildcard */*.frag)
COMP_SOURCES = $(wildcard */*.comp)
GLSL_INCLUDES = $(wildcard include/*.glsl)

VERT_SPV = $(VERT_SOURCES:%.vert=%_vert.spv)
FRAG_SPV = $(FRAG_SOURCES:%.frag=%_frag.spv)
COMP_SPV = $(COMP_SOURCES:%.comp=%_comp.spv)


GLSLC_FLAGS = -g -I include --target-env=vulkan1.2

all : $(VERT_SPV) $(FRAG_SPV) $(COMP_SPV)

$(VERT_SPV) : $(VERT_SOURCES) $(GLSL_INCLUDES)
	glslc $(GLSLC_FLAGS) $(@:%_vert.spv=%.vert) -o $@

$(FRAG_SPV) : $(FRAG_SOURCES) $(GLSL_INCLUDES)
	glslc $(GLSLC_FLAGS) $(@:%_frag.spv=%.frag) -o $@

$(COMP_SPV) : $(COMP_SOURCES) $(GLSL_INCLUDES)
	glslc $(GLSLC_FLAGS) $(@:%_comp.spv=%.comp) -o $@