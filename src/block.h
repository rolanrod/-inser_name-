#pragma once
#include <vector>
#include <map>
#include "position.h"
#include "colors.h"

class Block {
  private:
    int cellSize;
    
    std::vector<Color> colors;
    int rowOffset;
    int colOffset;

  public: 
    Block();
    void draw(int offsetX, int offsetY);
    void move(int rows, int cols);
    void rotate();
    void rotateBack();

    std::vector<Position> getCellPositions();
    int id;
    std::map<int, std::vector<Position>> cells;
    int rotState;
};