#include "game.h"
#include <random>
#include <iostream>

std::vector<Block> returnBlocks()
{
  return {IBlock(), JBlock(), OBlock(), ZBlock(), SBlock(), TBlock(), LBlock()};
}

bool Game::isBlockOutside()
{
  std::vector<Position> tiles = currentBlock.getCellPositions();
  for (auto tile : tiles)
  {
    if (grid.isOutside(tile.row, tile.col))
      return true;
  }
  return false;
}

Game::Game()
{
  srand(time(0));
  grid = Grid();
  blocks = returnBlocks();
  currentBlock = getRandomBlock();
  nextBlock = getRandomBlock();
  gameOver = false;
  score = 0;
}

Block Game::getRandomBlock()
{
  if (blocks.empty())
  {
    blocks = returnBlocks();
  }

  // Don't ask me why, but if I don't call rand() first like this,
  // randomizing doesn't work.
  for (int i = 0; i < 5; i++)
  {
    int j = (rand() % 100);
    while (j < 0)
      ;
  }

  int randomIndex = (rand() % blocks.size());
  Block block = blocks[randomIndex];
  blocks.erase(blocks.begin() + randomIndex);
  return block;
}

void Game::draw()
{
  {
    grid.draw();
    currentBlock.draw(1, 1);
    switch (nextBlock.id)
    {
    case 3:
      nextBlock.draw(255, 290);
      break;
    case 4:
      nextBlock.draw(255, 280);
      break;
    default:
      nextBlock.draw(270, 270);
      break;
    }
  }
}

void Game::handleInput()
{
  int keyPressed = GetKeyPressed();
  if (gameOver && keyPressed != 0)
  {
    gameOver = false;
    reset();
  }
  switch (keyPressed)
  {
  case KEY_LEFT:
    if (!gameOver)
    {
      currentBlock.move(0, -1);
      if (isBlockOutside() || !blockFits())
        currentBlock.move(0, 1);
    }
    break;

  case KEY_RIGHT:
    if (!gameOver)
    {
      currentBlock.move(0, 1);
      if (isBlockOutside() || !blockFits())
        currentBlock.move(0, -1);
    }
    break;
  case KEY_DOWN:
    if (!gameOver)
    {
      currentBlock.move(1, 0);
      if (isBlockOutside() || !blockFits())
        currentBlock.move(-1, 0);
    }
    updateScore(0, 1);
    break;

  case KEY_UP:
    if (!gameOver)
    {
      rotateBlock();
    }
    break;
  }
}

void Game::rotateBlock()
{
  currentBlock.rotate();
  if (isBlockOutside() || !blockFits())
    currentBlock.rotateBack();
}

void Game::lockBlock()
{
  std::vector<Position> tiles = currentBlock.getCellPositions();
  for (auto tile : tiles)
  {
    grid.grid[tile.row][tile.col] = currentBlock.id;
  }
  currentBlock = nextBlock;
  if (!blockFits())
  {
    gameOver = true;
    return;
  }
  nextBlock = getRandomBlock();
  int rowsCleared = grid.clearFullRows();
  updateScore(rowsCleared, 0);
}

bool Game::blockFits()
{
  std::vector<Position> tiles = currentBlock.getCellPositions();
  for (auto tile : tiles)
  {
    if (grid.isCellEmpty(tile.row, tile.col) == false)
      return false;
  }
  return true;
}

void Game::moveBlockDown()
{
  currentBlock.move(1, 0);
  if (isBlockOutside() || !blockFits())
  {
    currentBlock.move(-1, 0);
    lockBlock();
  }
}

void Game::reset()
{
  grid.initialize();
  blocks = returnBlocks();
  currentBlock = getRandomBlock();
  nextBlock = getRandomBlock();
  score = 0;
  gameOver = false;
}

void Game::updateScore(int lines, int moveDownPoints)
{
  switch (lines)
  {
  case 1:
    score += 100;
    break;
  case 2:
    score += 300;
    break;
  case 3:
    score += 500;
    break;
  case 4:
    score += 700;
    break;
  default:
    break;
  }
  score += moveDownPoints;
}