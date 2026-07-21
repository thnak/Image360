// Vendored stb_image (https://github.com/nothings/stb, public domain) -
// this is the one TU that compiles its implementation; every other
// #include "stb/stb_image.h" only sees the declarations. See
// ImageLoader::DecodeStandardImage/GetStandardImageDimensions.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
