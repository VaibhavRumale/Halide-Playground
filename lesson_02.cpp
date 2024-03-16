
#include "Halide.h"
#include <iostream>
#include <chrono>
#include "halide_image_io.h"
using namespace Halide::Tools;

int main(int argc, char **argv) {


    Halide::Buffer<uint8_t> input = load_image("images/rgb.PNG");
    std::cout << "Input image dimensions: " << input.width() << ", " << input.height() << ", " << input.channels() << std::endl;


     
    auto start = std::chrono::high_resolution_clock::now();


    Halide::Func brighter;


    Halide::Var x, y, c;

    Halide::Expr value = input(x, y, c);

    value = Halide::cast<float>(value);


    value = value * 1.5f;

    value = Halide::min(value, 255.0f);

    value = Halide::cast<uint8_t>(value);

    brighter(x, y, c) = value;


    Halide::Buffer<uint8_t> output =
        brighter.realize({input.width(), input.height(), input.channels()});
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "Image processing took " << duration.count() << " milliseconds." << std::endl;

    save_image(output, "brighter.PNG");
    std::cout << "Image saved successfully." << std::endl;

     

    printf("Success!\n");
    return 0;
}
