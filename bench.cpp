#include "Halide.h"
#include <stdio.h>
#include <chrono>
#include "halide_image_io.h"
#include<iostream>

using namespace Halide;

int main(int argc, char** argv) {
    // Define Vars
    Var x("x"), y("y"), c("c");

    // Load the input image
    Halide::Buffer<uint8_t> input = Tools:: load_image("images/rgb.PNG");
    std::cout<<"TESTING!!!!";
    // Create a Func for the input
    Func input_func("input_func");
    input_func(x, y, c) = input(x, y, c);

    // Define a 3x3 box blur filter
    Func box_blur("box_blur");
    box_blur(x, y, c) = (input_func(x - 1, y - 1, c) + input_func(x, y - 1, c) + input_func(x + 1, y - 1, c) +
                         input_func(x - 1, y, c) + input_func(x, y, c) + input_func(x + 1, y, c) +
                         input_func(x - 1, y + 1, c) + input_func(x, y + 1, c) + input_func(x + 1, y + 1, c)) / 9;

    // Create an output buffer with the same dimensions as the input
    Buffer<uint8_t> output(input.width(), input.height(), 3);

    // Realize the output
    box_blur.realize(output);

    auto start = std::chrono::high_resolution_clock::now();

    // Save the output image
   Tools::save_image(output, "output.png"); 

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "Image processing took " << duration.count() << " milliseconds." << std::endl;

    return 0;
}
