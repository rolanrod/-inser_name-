#include "block.h"

Block::Block()
{
  cellSize = 30;
  rotState = 0;
  colors = getCellColors();
  rowOffset = 0;
  colOffset = 0;
}

void Block::draw(int offsetX, int offsetY)
{
  std::vector<Position> tiles = getCellPositions();

  for (Position item : tiles)
  {
    DrawRectangle(item.col * cellSize + offsetX, item.row * cellSize + offsetY,
                  cellSize - 1, cellSize - 1, colors[id]);
  }
}

void Block::move(int rows, int cols)
{
  rowOffset += rows;
  colOffset += cols;
}

void Block::rotate() { 
  rotState++;
  if (rotState == (int)cells.size()) rotState = 0;
}

void Block::rotateBack(){
  rotState--;
  if (rotState == -1) rotState = (int)cells.size() - 1;
}

std::vector<Position> Block::getCellPositions()
{
  std::vector<Position> tiles = cells[rotState];
  std::vector<Position> movedTiles;
  for (auto &item : tiles)
  {
    Position newPos = Position(item.row + rowOffset, item.col + colOffset);
    movedTiles.push_back(newPos);
  }
  return movedTiles;
}