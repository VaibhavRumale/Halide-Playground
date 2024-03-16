

#include "Halide.h"
#include <algorithm>
#include <chrono>
#include <stdio.h>
using namespace Halide;

int main(int argc, char **argv) {

 
    auto start = std::chrono::high_resolution_clock::now();


    Var x("x"), y("y");

    {
        Func gradient("gradient");
        gradient(x, y) = x + y;
        gradient.trace_stores();

        printf("Evaluating gradient row-major\n");
        std::vector<uint8_t> output_image(100 * 100 * 8);

        Buffer<int> output = gradient.realize({4, 4});
 
        printf("Pseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("\n");

    }

  
    {
        Func gradient("gradient_col_major");
        gradient(x, y) = x + y;
        gradient.trace_stores();


        gradient.reorder(y, x);


        printf("Evaluating gradient column-major\n");
        Buffer<int> output = gradient.realize({4, 4});


         

        printf("Pseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("\n");
    }


    {
        Func gradient("gradient_split");
        gradient(x, y) = x + y;
        gradient.trace_stores();


        Var x_outer, x_inner;
        gradient.split(x, x_outer, x_inner, 2);



        printf("Evaluating gradient with x split into x_outer and x_inner \n");
        Buffer<int> output = gradient.realize({4, 4});


        printf("Pseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("\n");

    }

    {
        Func gradient("gradient_fused");
        gradient(x, y) = x + y;

     
        Var fused;
        gradient.fuse(x, y, fused);

        printf("Evaluating gradient with x and y fused\n");
        Buffer<int> output = gradient.realize({4, 4});



        printf("Pseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("\n");
    }

    {
        Func gradient("gradient_tiled");
        gradient(x, y) = x + y;
        gradient.trace_stores();

  
        Var x_outer, x_inner, y_outer, y_inner;
        gradient.split(x, x_outer, x_inner, 4);
        gradient.split(y, y_outer, y_inner, 4);
        gradient.reorder(x_inner, y_inner, x_outer, y_outer);


        printf("Evaluating gradient in 4x4 tiles\n");
        Buffer<int> output = gradient.realize({8, 8});


        printf("Pseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("\n");
    }

    {
        Func gradient("gradient_in_vectors");
        gradient(x, y) = x + y;
        gradient.trace_stores();

\
        Var x_outer, x_inner;
        gradient.split(x, x_outer, x_inner, 4);
        gradient.vectorize(x_inner);


        printf("Evaluating gradient with x_inner vectorized \n");
        Buffer<int> output = gradient.realize({8, 4});


        printf("Pseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("\n");
    }

    {
        Func gradient("gradient_unroll");
        gradient(x, y) = x + y;
        gradient.trace_stores();

        Var x_outer, x_inner;
        gradient.split(x, x_outer, x_inner, 2);
        gradient.unroll(x_inner);



        printf("Evaluating gradient unrolled by a factor of two\n");
        Buffer<int> result = gradient.realize({4, 4});


        printf("Pseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("\n");
    }

    {
        Func gradient("gradient_split_7x2");
        gradient(x, y) = x + y;
        gradient.trace_stores();

        Var x_outer, x_inner;
        gradient.split(x, x_outer, x_inner, 3);

        printf("Evaluating gradient over a 7x2 box with x split by three \n");
        Buffer<int> output = gradient.realize({7, 2});


        printf("Pseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("\n");

    }

    {


        Func gradient("gradient_fused_tiles");
        gradient(x, y) = x + y;
        gradient.trace_stores();

        Var x_outer, y_outer, x_inner, y_inner, tile_index;
        gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);
        gradient.fuse(x_outer, y_outer, tile_index);
        gradient.parallel(tile_index);

        printf("Evaluating gradient tiles in parallel\n");
        Buffer<int> output = gradient.realize({8, 8});


        printf("Pseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("\n");
    }

    {
        Func gradient_fast("gradient_fast");
        gradient_fast(x, y) = x + y;

        Var x_outer, y_outer, x_inner, y_inner, tile_index;
        gradient_fast
            .tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
            .fuse(x_outer, y_outer, tile_index)
            .parallel(tile_index);

    
        Var x_inner_outer, y_inner_outer, x_vectors, y_pairs;
        gradient_fast
            .tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, 4, 2)
            .vectorize(x_vectors)
            .unroll(y_pairs);


        Buffer<int> result = gradient_fast.realize({350, 250});


        printf("Pseudo-code for the schedule:\n");
        gradient_fast.print_loop_nest();
        printf("\n");

    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "TIME " << duration.count() << " milliseconds." << std::endl;
    printf("Success!\n");
    return 0;
}
