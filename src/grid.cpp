#include "grid.h"
#include "colors.h"
#include <iostream>

Grid::Grid()
{
  numRows = 20;
  numCols = 10;
  cellSize = 30;
  initialize();
  colors = getCellColors();
}

void Grid::initialize()
{
  for (int row = 0; row < numRows; row++)
  {
    for (int column = 0; column < numCols; column++)
    {
      grid[row][column] = 0;
    }
  }
}

void Grid::print()
{
  for (const auto &row : grid)
  {
    for (int elem : row)
    {
      std::cout << elem << " ";
    }
    std::cout << "\n";
  }
}

void Grid::draw()
{
  for (int row = 0; row < numRows; row++){
    for (int col = 0; col < numCols; col++){
      int cellVal = grid[row][col];
      DrawRectangle(col * cellSize + 1, row * cellSize + 1, cellSize - 1 , cellSize - 1, colors[cellVal]);
    }
  }
}

bool Grid::isOutside(int row, int col) {return !(row >= 0 && row < numRows && col >= 0 && col < numCols);}

bool Grid::isCellEmpty(int row, int col){return (grid[row][col] == 0);}

bool Grid::isRowFull(int row)
{
  for (int col = 0; col < numCols; col++){
    if (isCellEmpty(row, col)) return false;
  }
  return true;
}

void Grid::moveRowDown(int row, int numRows)
{
  for (int col = 0; col < numCols; col++){
    grid[row + numRows][col] = grid[row][col];
    grid[row][col] = 0;
    }
}

void Grid::clearRow(int row){
  for (int col = 0; col < numCols; col++){
    grid[row][col] = 0;
  }
}

int Grid::clearFullRows(){
  int full = 0;
  for (int row = numRows - 1; row >= 0; row--){
    if (isRowFull(row)){
      clearRow(row);
      full++;
    }
    else if (full > 0){
      moveRowDown(row, full);
    }
  }
  return full;
}