#pragma once
#include <vector>
#include "raylib.h"

class Grid
{
private:
  int numRows;
  int numCols;
  int cellSize;
  std::vector<Color> colors;
  bool isRowFull(int row);
  void moveRowDown(int row, int numRows);
  void clearRow(int row);

public:
  Grid();
  int grid[20][10];
  void initialize();
  void print();
  void draw();
  bool isOutside(int row, int col);
  bool isCellEmpty(int row, int col);
  int clearFullRows();
};