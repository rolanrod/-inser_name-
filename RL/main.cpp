#include <torch/torch.h>
#include <iostream>
#include "raylib.h"

int main() {
  torch::Tensor tensor = torch::rand({3, 3});
  std::cout << tensor << std::endl;
  InitWindow(500, 600, "Tetris");
   while (!WindowShouldClose()) {

   }

   CloseWindow();
}
