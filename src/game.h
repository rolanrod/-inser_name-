#pragma once
#include "grid.h"
#include "blocks.h"

class Game
{
private:
  bool isBlockOutside();
  std::vector<Block> blocks;
  Block currentBlock;
  Block nextBlock;
  void rotateBlock();
  void lockBlock();
  bool blockFits();
  void updateScore(int lines, int moveDownPoints);

public:
  Game();
  Block getRandomBlock();
  void draw();
  Grid grid;
  void handleInput();
  void moveBlockDown();
  bool gameOver;
  int score;
  void reset();
};